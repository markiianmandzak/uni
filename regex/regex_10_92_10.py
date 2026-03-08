import pandas as pd
import re
import argparse
from tqdm import tqdm
import os

DATA_ROOT = "data/feature-normalization-hackathon/data"

def parse_taxonomy(tax_df):
    print("1. Building Precision-Engineered Ruleset...")
    taxonomy_rules = {}

    # Strict dimensional anchors
    dim_anchors = {
        'Länge': r'\b(?:länge|l\s*[:=])',
        'Breite': r'\b(?:breite|b\s*[:=])',
        'Durchmesser': r'\b(?:durchmesser|ø|d\s*[:=])',
        'Höhe': r'\b(?:höhe|h\s*[:=])',
    }

    for _, row in tax_df.iterrows():
        cat = row['category']
        feat = row['feature_name']
        f_type = row['feature_type']
        raw_vals = str(row['aggregated_feature_values'])
        extracted_vals = re.findall(r"\[(.*?)\]", raw_vals)
        if not extracted_vals: continue
            
        if f_type == 'categorical':
            # FIX 1: Sort by length DESCENDING is mandatory to catch "Edelstahl (A2)" before "Edelstahl"
            valid_strings = sorted(extracted_vals, key=len, reverse=True)
            
            # FIX 2: Negative Lookahead/Lookbehind to prevent matching "Stahl" inside "Chrom-Vanadium-Stahl"
            # We allow spaces/slashes but NOT hyphens or letters immediately touching the word
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
                # FIX 3: Catch 1.000 by allowing dots in the middle of digits
                regex_str = r"\b(\d{1,3}(?:\.\d{3})*|\d+)\s*(Stück|Stk\.?|Rolle|Karton|Pack|Paar|Set)\b"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_vpe',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'unit_mapping': {'stk': 'Stück', 'stk.': 'Stück'} 
                }
            elif feat in dim_anchors:
                unit_match = re.search(r'[A-Za-z%²³"\'°]+$', first_val)
                unit = unit_match.group(0) if unit_match else ""
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

    tax_df = pd.read_parquet(os.path.join(DATA_ROOT, 'taxonomy/taxonomy.parquet'))
    taxonomy_rules = parse_taxonomy(tax_df)
    prod_df = pd.read_parquet(os.path.join(DATA_ROOT, args.split, 'products.parquet'))
    
    if args.split == 'test':
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, 'test/submission.parquet'))
    else:
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, 'val/product_features.parquet'))
        target_df['feature_value'] = None 
    
    merged_df = target_df.merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    
    predicted_values = []
    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        rule = taxonomy_rules.get((row.category, row.feature_name))
        extracted_val = None
        
        if rule:
            # We combine title and desc but keep them separate for priority
            title = str(row.title)
            desc = str(row.description)
            full_text = title + " " + desc
            
            if rule['type'] == 'categorical':
                # Check for all matches. If multiple unique values exist, we ABSTAIN.
                matches = rule['pattern'].findall(full_text)
                if matches:
                    unique_matches = set([m.lower() for m in matches])
                    if len(unique_matches) == 1:
                        val_lower = list(unique_matches)[0]
                        # Return the specific casing from taxonomy
                        for v in rule['values']:
                            if v.lower() == val_lower:
                                extracted_val = v
                                break
                                
            elif rule['type'] == 'numeric_dim':
                match = rule['pattern'].search(full_text)
                if match:
                    num = match.group(1).replace(',', '.') 
                    extracted_val = f"{num} {rule['unit']}".strip() if " " in rule['format_example'] else f"{num}{rule['unit']}".strip()
                    
            elif rule['type'] == 'numeric_thread':
                match = rule['pattern'].search(full_text)
                if match:
                    raw = match.group(1).upper().replace(" ", "")
                    for ex in rule['examples']:
                        if raw == ex.upper().replace(" ", ""):
                            extracted_val = ex
                            break
                            
            elif rule['type'] == 'numeric_vpe':
                match = rule['pattern'].search(full_text)
                if match:
                    # Remove the thousands separator dot (1.000 -> 1000)
                    clean_num = match.group(1).replace('.', '')
                    unit = rule['unit_mapping'].get(match.group(2).lower(), match.group(2).title())
                    extracted_val = f"{clean_num} {unit}"
                            
        predicted_values.append(extracted_val)
        
    target_df['feature_value'] = predicted_values
    target_df[['uid', 'feature_name', 'feature_value', 'feature_type']].to_parquet(f'submission_{args.split}.parquet', index=False)

if __name__ == "__main__":
    main()