import pandas as pd
import re
import argparse
from tqdm import tqdm
import os

DATA_ROOT = "data/feature-normalization-hackathon/data"

def parse_taxonomy(tax_df):
    print("1. Building Gold-Standard Ruleset (Precision > 95% Focus)...")
    taxonomy_rules = {}

    dim_anchors = {
        'Länge': r'(?:länge|l\s*[:=])',
        'Breite': r'(?:breite|b\s*[:=])',
        'Gesamtbreite': r'(?:gesamtbreite|breite|b\s*[:=])',
        'Höhe': r'(?:höhe|h\s*[:=])',
        'Gesamthöhe': r'(?:gesamthöhe|höhe|h\s*[:=])',
        'Durchmesser': r'(?:durchmesser|ø|d\s*[:=])',
        'Außen-Ø': r'(?:außen-ø|außen\s*ø|da\s*[:=])',
        'Innen-Ø': r'(?:innen-ø|innen\s*ø|di\s*[:=])'
    }

    # Added new toxic features from your latest error report
    toxic_features = {
        'Frontfarbe', 'Korpusfarbe', 'Wiedergabeformate', 'Plattentyp', 
        'Stationäre Phase', 'Körnung', 'Felgenmaterial', 'Schneidstoff', 
        'Laufbelag', 'Hülsenmaterial', 'Toleranzklasse', 'Gasart', 
        'Brenngas', 'Für Modell', 'Rollenmaterial', 'Stiel', 'Gebindeart'
    }

    for _, row in tax_df.iterrows():
        cat = row['category']
        feat = row['feature_name']
        f_type = row['feature_type']
        
        if feat in toxic_features:
            continue

        raw_vals = str(row['aggregated_feature_values'])
        extracted_vals = re.findall(r"\[(.*?)\]", raw_vals)
        if not extracted_vals:
            continue
            
        if f_type == 'categorical':
            valid_strings = sorted(extracted_vals, key=len, reverse=True)
            pattern_str = r'\b(' + '|'.join(map(re.escape, valid_strings)) + r')\b'
            taxonomy_rules[(cat, feat)] = {
                'type': 'categorical',
                'values': valid_strings,
                'pattern': re.compile(pattern_str, re.IGNORECASE)
            }
            
        elif f_type == 'numeric':
            first_val = extracted_vals[0]
            if feat in ['Gewinde-Ø', 'Antrieb']:
                # Added word boundaries and restricted lookahead to prevent matching model numbers
                regex_str = r"\b(M\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?/\d+\s*\"?|G\s*\d+(?:[.,]\d+)?/\d+\s*\"?)\b"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_thread',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'examples': extracted_vals
                }
            elif feat == 'Verpackungseinheit':
                regex_str = r"\b(\d+)\s*(Stück|Stk\.?|Rolle|Karton|Pack|Paar|Set)\b"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_vpe',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'unit_mapping': {'stk': 'Stück', 'stk.': 'Stück'} 
                }
            elif feat in dim_anchors:
                unit_match = re.search(r'[A-Za-z%²³"\'°]+$', first_val)
                unit = unit_match.group(0) if unit_match else ""
                anchor = dim_anchors[feat]
                # Anchor must be very close to the number
                regex_str = r"\b" + anchor + r"\s*(\d+(?:[.,]\d+)?)\s*(" + re.escape(unit) + r")\b"
                
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
    taxonomy_rules = parse_taxonomy(tax_df)
    
    prod_df = pd.read_parquet(os.path.join(DATA_ROOT, args.split, 'products.parquet'))
    
    if args.split == 'test':
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, 'test', 'submission.parquet'))
    else:
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, 'val', 'product_features.parquet'))
        target_df['feature_value'] = None 
    
    merged_df = target_df.merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    
    predicted_values = []
    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        rule = taxonomy_rules.get((row.category, row.feature_name))
        extracted_val = None
        
        if rule:
            # TITLE PRIORITY: Title is usually cleaner than Description
            title = str(row.title).strip()
            desc = str(row.description).strip()
            
            if rule['type'] == 'categorical':
                # Check Title first
                matches = rule['pattern'].findall(title)
                if not matches: # Fallback to description
                    matches = rule['pattern'].findall(desc)
                
                if matches:
                    unique = set([m.lower() for m in matches])
                    if len(unique) == 1:
                        val_lower = unique.pop()
                        for v in rule['values']:
                            if v.lower() == val_lower:
                                extracted_val = v
                                break
                                
            elif rule['type'] == 'numeric_dim':
                # For dimensions, ONLY trust the title match for ultra-high precision
                match = rule['pattern'].search(title)
                if not match: match = rule['pattern'].search(desc)
                if match:
                    num = match.group(1).replace(',', '.') 
                    u = rule['unit']
                    extracted_val = f"{num} {u}".strip() if " " in rule['format_example'] else f"{num}{u}".strip()
                    
            elif rule['type'] == 'numeric_thread':
                match = rule['pattern'].search(title)
                if not match: match = rule['pattern'].search(desc)
                if match:
                    raw = match.group(1).upper().replace(" ", "")
                    for ex in rule['examples']:
                        if raw == ex.upper().replace(" ", ""):
                            extracted_val = ex
                            break
                            
            elif rule['type'] == 'numeric_vpe':
                match = rule['pattern'].search(title) or rule['pattern'].search(desc)
                if match:
                    extracted_val = f"{match.group(1)} {rule['unit_mapping'].get(match.group(2).lower(), match.group(2).title())}"
                            
        predicted_values.append(extracted_val)
        
    target_df['feature_value'] = predicted_values
    target_df[['uid', 'feature_name', 'feature_value', 'feature_type']].to_parquet(f'submission_{args.split}.parquet', index=False)

if __name__ == "__main__":
    main()