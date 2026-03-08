import pandas as pd
import re
import argparse
from tqdm import tqdm
import os

DATA_ROOT = "data/feature-normalization-hackathon/data"

def parse_taxonomy(tax_df):
    print("1. Building Surgical High-Precision Ruleset (Toxic Features Blocked)...")
    taxonomy_rules = {}

    # Strict anchors for safe dimension extraction
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

    # Features that have >25% error rate because of split-parts or complex formatting
    toxic_categorical_features = {
        'Frontfarbe', 'Korpusfarbe', 'Wiedergabeformate', 'Plattentyp', 
        'Stationäre Phase', 'Körnung', 'Felgenmaterial', 'Schneidstoff', 
        'Laufbelag', 'Hülsenmaterial'
    }

    for _, row in tax_df.iterrows():
        cat = row['category']
        feat = row['feature_name']
        f_type = row['feature_type']
        raw_vals = str(row['aggregated_feature_values'])
        
        # SKIP toxic features immediately so they fall back to None
        if feat in toxic_categorical_features:
            continue

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
            
            # Safe Thread Sniper (Restricted to proven features)
            if feat in ['Gewinde-Ø', 'Antrieb']:
                regex_str = r"\b(M\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?/\d+\s*\"?|G\s*\d+(?:[.,]\d+)?/\d+\s*\"?)(?!\w)"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_thread',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'examples': extracted_vals
                }
                
            # Safe Packaging Units (Removed 'Inhalt')
            elif feat == 'Verpackungseinheit':
                regex_str = r"\b(\d+)\s*(Stück|Stk\.?|Rolle|Karton|Pack|Paar|Set|m|kg)\b"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_vpe',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'unit_mapping': {'stk': 'Stück', 'stk.': 'Stück'} 
                }
                
            # Safe Anchored Dimensions
            elif feat in dim_anchors:
                parts = first_val.split()
                if len(parts) > 1 and not bool(re.search(r'\d', parts[-1])):
                    unit = parts[-1]
                else:
                    unit_match = re.search(r'[A-Za-z%²³"\'°]+$', first_val)
                    unit = unit_match.group(0) if unit_match else ""
                
                anchor = dim_anchors[feat]
                regex_str = r"\b" + anchor + r"\s*(\d+(?:[.,]\d+)?)\s*(" + re.escape(unit) + r")(?!\w)"
                
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

    tax_path = os.path.join(DATA_ROOT, 'taxonomy', 'taxonomy.parquet')
    tax_df = pd.read_parquet(tax_path)
    taxonomy_rules = parse_taxonomy(tax_df)
    
    prod_path = os.path.join(DATA_ROOT, args.split, 'products.parquet')
    prod_df = pd.read_parquet(prod_path)
    
    if args.split == 'test':
        target_path = os.path.join(DATA_ROOT, 'test', 'submission.parquet')
        target_df = pd.read_parquet(target_path)
    else:
        target_path = os.path.join(DATA_ROOT, 'val', 'product_features.parquet')
        target_df = pd.read_parquet(target_path)
        target_df['feature_value'] = None 
    
    merged_df = target_df.merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    merged_df['text'] = (merged_df['title'].fillna('') + " " + merged_df['description'].fillna('')).astype(str)
    
    predicted_values = []
    
    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        cat = row.category
        feat = row.feature_name
        text = row.text
        
        rule = taxonomy_rules.get((cat, feat))
        extracted_val = None
        
        if rule and text.strip():
            
            if rule['type'] == 'categorical':
                all_matches = rule['pattern'].findall(text)
                if all_matches:
                    unique_matches_lower = set([m.lower() for m in all_matches])
                    # Uniqueness constraint for safety
                    if len(unique_matches_lower) == 1:
                        matched_str_lower = unique_matches_lower.pop()
                        for v in rule['values']:
                            if v.lower() == matched_str_lower:
                                extracted_val = v
                                break
                                
            elif rule['type'] == 'numeric_dim':
                match = rule['pattern'].search(text)
                if match:
                    number_part = match.group(1).replace(',', '.') 
                    unit_part = rule['unit']
                    has_space = " " in rule['format_example']
                    extracted_val = f"{number_part} {unit_part}".strip() if has_space else f"{number_part}{unit_part}".strip()
                    
            elif rule['type'] == 'numeric_thread':
                match = rule['pattern'].search(text)
                if match:
                    raw_match = match.group(1).upper().replace(" ", "")
                    for ex in rule['examples']:
                        if raw_match == ex.upper().replace(" ", ""):
                            extracted_val = ex
                            break
                    if not extracted_val:
                        if raw_match.startswith('M') and len(raw_match) > 1:
                            extracted_val = f"M {raw_match[1:]}"
                        else:
                            extracted_val = match.group(1)
                            
            elif rule['type'] == 'numeric_vpe':
                match = rule['pattern'].search(text)
                if match:
                    number = match.group(1)
                    raw_unit = match.group(2).title() 
                    unit = rule['unit_mapping'].get(raw_unit.lower(), raw_unit)
                    extracted_val = f"{number} {unit}"
                            
        predicted_values.append(extracted_val)
        
    target_df['feature_value'] = predicted_values
    
    output_filename = f'submission_{args.split}.parquet'
    final_submission = target_df[['uid', 'feature_name', 'feature_value', 'feature_type']]
    final_submission.to_parquet(output_filename, index=False)

if __name__ == "__main__":
    main()