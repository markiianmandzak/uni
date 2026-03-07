import pandas as pd
import re
import argparse
from tqdm import tqdm
import os

DATA_ROOT = "data/feature-normalization-hackathon/data"

def optimize_taxonomy_with_train(tax_df):
    print("1. Building Precision-Engineered Ruleset (Addressing Substrings & Anchors)...")
    taxonomy_rules = {}

    # FEATURES TO IGNORE IN TIER 1 (Too much context required)
    toxic_features = {
        'Frontfarbe', 'Korpusfarbe', 'Wiedergabeformate', 'Plattentyp', 
        'Stationäre Phase', 'Felgenmaterial', 'Schneidstoff', 
        'Laufbelag', 'Hülsenmaterial', 'Toleranzklasse', 'Gasart', 
        'Brenngas', 'Für Modell', 'Rollenmaterial', 'Stiel', 'Gebindeart'
    }

    # Strict dimensional anchors
    dim_anchors = {
        'Länge': r'\b(?:länge|l\s*[:=])',
        'Breite': r'\b(?:breite|b\s*[:=])',
        'Gesamtbreite': r'\b(?:gesamtbreite|breite|b\s*[:=])',
        'Höhe': r'\b(?:höhe|h\s*[:=])',
        'Gesamthöhe': r'\b(?:gesamthöhe|höhe|h\s*[:=])',
        'Durchmesser': r'\b(?:durchmesser|ø|d\s*[:=])',
        'Außen-Ø': r'\b(?:außen-ø|da\s*[:=])',
        'Innen-Ø': r'\b(?:innen-ø|di\s*[:=])',
        'Körnung': r'\b(?:körnung|k\s*[:=-]?)' # FIX: Added Körnung anchor
    }

    for _, row in tax_df.iterrows():
        cat = row['category']
        feat = row['feature_name']
        f_type = row['feature_type']
        
        if feat in toxic_features:
            continue

        raw_vals = str(row['aggregated_feature_values'])
        extracted_vals = re.findall(r"\[(.*?)\]", raw_vals)
        if not extracted_vals: continue
            
        if f_type == 'categorical':
            # Sort by length DESCENDING to prioritize longer matches
            valid_strings = sorted(extracted_vals, key=len, reverse=True)
            
            # Use negative lookbehind/lookahead to avoid matching substrings inside compound words
            # Example: Will not match "Stahl" inside "Chrom-Vanadium-Stahl"
            pattern_str = r'(?<![\w-])(' + '|'.join(map(re.escape, valid_strings)) + r')(?![\w-])'
            
            taxonomy_rules[(cat, feat)] = {
                'type': 'categorical',
                'values': valid_strings,
                'pattern': re.compile(pattern_str, re.IGNORECASE)
            }
            
        elif f_type == 'numeric':
            first_val = extracted_vals[0]
            if feat in ['Gewinde-Ø', 'Antrieb']:
                regex_str = r"\b(M\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?/\d+\s*\"?|G\s*\d+(?:[.,]\d+)?/\d+\s*\"?)\b"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_thread',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'examples': extracted_vals
                }
            elif feat == 'Verpackungseinheit':
                # FIX: Handle 1.000 format
                regex_str = r"\b(\d{1,3}(?:\.\d{3})*|\d+)\s*(Stück|Stk\.?|Rolle|Karton|Pack|Paar|Set)\b"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_vpe',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'unit_mapping': {'stk': 'Stück', 'stk.': 'Stück'} 
                }
            elif feat in dim_anchors:
                unit_match = re.search(r'[A-Za-z%²³"\'°]+$', first_val)
                unit = unit_match.group(0) if unit_match else ""
                # Mandatory Anchor + Number + Unit
                regex_str = dim_anchors[feat] + r"\s*(\d+(?:[.,]\d+)?)\s*(" + re.escape(unit) + r")\b"
                
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_dim',
                    'unit': unit,
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'format_example': first_val
                }

    return taxonomy_rules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val')
    args = parser.parse_args()

    tax_df = pd.read_parquet(os.path.join(DATA_ROOT, 'taxonomy', 'taxonomy.parquet'))
    taxonomy_rules = optimize_taxonomy_with_train(tax_df)
    
    prod_df = pd.read_parquet(os.path.join(DATA_ROOT, args.split, 'products.parquet'))
    
    if args.split == 'test':
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, 'test', 'submission.parquet'))
    else:
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, 'val', 'product_features.parquet'))
        target_df['feature_value'] = None 
    
    merged_df = target_df.merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    
    predicted_values = []
    print(f"2. Processing {args.split.upper()} split...")
    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        rule = taxonomy_rules.get((row.category, row.feature_name))
        extracted_val = None
        
        if rule:
            title = str(row.title).strip()
            desc = str(row.description).strip()
            
            if rule['type'] == 'categorical':
                # FIX: Check Title FIRST. If it has a clean, unique match, STOP. Don't look at the messy description.
                title_matches = rule['pattern'].findall(title)
                if title_matches:
                    unique_matches = set([m.lower() for m in title_matches])
                    if len(unique_matches) == 1:
                        val_lower = list(unique_matches)[0]
                        for v in rule['values']:
                            if v.lower() == val_lower:
                                extracted_val = v
                                break
                
                # Only if the title fails do we check the description, and we still require uniqueness.
                if not extracted_val:
                    desc_matches = rule['pattern'].findall(desc)
                    if desc_matches:
                        unique_matches = set([m.lower() for m in desc_matches])
                        if len(unique_matches) == 1:
                            val_lower = list(unique_matches)[0]
                            for v in rule['values']:
                                if v.lower() == val_lower:
                                    extracted_val = v
                                    break
                                
            elif rule['type'] == 'numeric_dim':
                # Dimensions are safer to grab from anywhere if the strict anchor is present
                full_text = title + " " + desc
                match = rule['pattern'].search(full_text)
                if match:
                    num = match.group(1).replace(',', '.') 
                    u = rule['unit']
                    extracted_val = f"{num} {u}".strip() if " " in rule['format_example'] else f"{num}{u}".strip()
                    
            elif rule['type'] == 'numeric_thread':
                full_text = title + " " + desc
                match = rule['pattern'].search(full_text)
                if match:
                    raw = match.group(1).upper().replace(" ", "")
                    for ex in rule['examples']:
                        if raw == ex.upper().replace(" ", ""):
                            extracted_val = ex
                            break
                            
            elif rule['type'] == 'numeric_vpe':
                full_text = title + " " + desc
                match = rule['pattern'].search(full_text)
                if match:
                    clean_num = match.group(1).replace('.', '') # Clean thousand separator
                    unit = rule['unit_mapping'].get(match.group(2).lower(), match.group(2).title())
                    extracted_val = f"{clean_num} {unit}"
                            
        predicted_values.append(extracted_val)
        
    target_df['feature_value'] = predicted_values
    target_df[['uid', 'feature_name', 'feature_value', 'feature_type']].to_parquet(f'submission_{args.split}.parquet', index=False)
    print("Done.")

if __name__ == "__main__":
    main()