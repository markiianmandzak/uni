import pandas as pd
import re
import argparse
from tqdm import tqdm
import os

# Set this to your actual data path
DATA_ROOT = "data/feature-normalization-hackathon/data"

def parse_taxonomy(tax_df):
    print("1. Building Conservative Taxonomy Ruleset...")
    taxonomy_rules = {}
    
    # Dimension keywords to anchor our search and prevent grabbing the wrong number
    dim_keywords = {
        'Länge': ['länge', 'l', 'len'],
        'Breite': ['breite', 'b', 'wid'],
        'Gesamtbreite': ['gesamtbreite', 'breite', 'b'],
        'Höhe': ['höhe', 'h', 'hei'],
        'Gesamthöhe': ['gesamthöhe', 'höhe', 'h'],
        'Durchmesser': ['durchmesser', 'ø', 'd', 'dia'],
        'Außen-Ø': ['außen-ø', 'außen', 'ø', 'd'],
        'Innen-Ø': ['innen-ø', 'innen', 'ø', 'd']
    }

    for _, row in tax_df.iterrows():
        cat = row['category']
        feat = row['feature_name']
        f_type = row['feature_type']
        raw_vals = str(row['aggregated_feature_values'])
        
        extracted_vals = re.findall(r"\[(.*?)\]", raw_vals)
        if not extracted_vals:
            continue
            
        if f_type == 'categorical':
            valid_strings = sorted(extracted_vals, key=len, reverse=True)
            
            # Conservative Categorical Matcher
            # For specific features where German compounds are common (e.g., "Edelstahlgehäuse" -> "Edelstahl")
            # We allow preceding letters but enforce a word boundary at the end so it doesn't match a prefix.
            if feat in ['Material', 'Farbe', 'Oberfläche', 'Ausführung']:
                pattern_str = r'\b[A-Za-zÄÖÜäöüß]*(' + '|'.join(map(re.escape, valid_strings)) + r')\b'
            else:
                pattern_str = r'\b(' + '|'.join(map(re.escape, valid_strings)) + r')\b'
                
            taxonomy_rules[(cat, feat)] = {
                'type': 'categorical',
                'values': valid_strings,
                'pattern': re.compile(pattern_str, re.IGNORECASE)
            }
            
        elif f_type == 'numeric':
            first_val = extracted_vals[0]
            
            # Heuristic: The Thread Sniper
            if 'Gewinde' in feat or 'Antrieb' in feat:
                regex_str = r"(M\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?/\d+\s*\"?|G\s*\d+(?:[.,]\d+)?/\d+\s*\"?)"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_thread',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'examples': extracted_vals
                }
                
            # Heuristic: Packaging Unit
            elif feat == 'Verpackungseinheit':
                regex_str = r"(\d+)\s*(Stück|Stk|Rolle|Karton|Pack|Paar|Set|m|kg)"
                taxonomy_rules[(cat, feat)] = {
                    'type': 'numeric_vpe',
                    'pattern': re.compile(regex_str, re.IGNORECASE),
                    'unit_mapping': {'stk': 'Stück', 'stk.': 'Stück'} 
                }
                
            # Heuristic: Dimensions and General Numerics
            else:
                parts = first_val.split()
                if len(parts) > 1 and not bool(re.search(r'\d', parts[-1])):
                    unit = parts[-1]
                else:
                    unit_match = re.search(r'[A-Za-z%²³"\'°]+$', first_val)
                    unit = unit_match.group(0) if unit_match else ""
                
                if feat in dim_keywords:
                    anchors = '|'.join(dim_keywords[feat])
                    # Pattern: Optional anchor word -> optional space/chars -> number -> unit
                    # The fixed line:
                    regex_str = r"(?:\b(" + anchors + r")\b.{0,10}?)?(\d+(?:[.,]\d+)?)\s*(" + re.escape(unit) + r")"
                    
                    taxonomy_rules[(cat, feat)] = {
                        'type': 'numeric_dim',
                        'unit': unit,
                        'pattern': re.compile(regex_str, re.IGNORECASE),
                        'format_example': first_val 
                    }
                else:
                    regex_str = r"(\d+(?:[.,]\d+)?)\s*(" + re.escape(unit) + r")"
                    
                    taxonomy_rules[(cat, feat)] = {
                        'type': 'numeric',
                        'unit': unit,
                        'pattern': re.compile(regex_str, re.IGNORECASE),
                        'format_example': first_val 
                    }
                
    return taxonomy_rules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val', 
                        help='Which split to run the extraction on')
    args = parser.parse_args()

    # 1. Load Taxonomy
    tax_path = os.path.join(DATA_ROOT, 'taxonomy', 'taxonomy.parquet')
    tax_df = pd.read_parquet(tax_path)
    taxonomy_rules = parse_taxonomy(tax_df)
    
    # 2. Load Products and Template
    print(f"2. Loading {args.split.upper()} Data...")
    prod_path = os.path.join(DATA_ROOT, args.split, 'products.parquet')
    prod_df = pd.read_parquet(prod_path)
    
    if args.split == 'test':
        target_path = os.path.join(DATA_ROOT, 'test', 'submission.parquet')
        target_df = pd.read_parquet(target_path)
    else:
        target_path = os.path.join(DATA_ROOT, 'val', 'product_features.parquet')
        target_df = pd.read_parquet(target_path)
        target_df['feature_value'] = None 
    
    print("3. Merging Text Context...")
    merged_df = target_df.merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    merged_df['text'] = (merged_df['title'].fillna('') + " " + merged_df['description'].fillna('')).astype(str)
    
    print("4. Executing Fast Extraction Loop...")
    predicted_values = []
    
    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        cat = row.category
        feat = row.feature_name
        text = row.text
        
        rule = taxonomy_rules.get((cat, feat))
        extracted_val = None
        
        if rule and text.strip():
            match = rule['pattern'].search(text)
            if match:
                if rule['type'] == 'categorical':
                    matched_str_lower = match.group(1).lower()
                    for v in rule['values']:
                        if v.lower() == matched_str_lower:
                            extracted_val = v
                            break
                            
                elif rule['type'] == 'numeric':
                    number_part = match.group(1).replace(',', '.') 
                    unit_part = rule['unit']
                    has_space = " " in rule['format_example']
                    extracted_val = f"{number_part} {unit_part}".strip() if has_space else f"{number_part}{unit_part}".strip()
                    
                elif rule['type'] == 'numeric_dim':
                    # Group 2 is the number because Group 1 is the optional anchor word
                    number_part = match.group(2)
                    if number_part:
                        number_part = number_part.replace(',', '.')
                        unit_part = rule['unit']
                        has_space = " " in rule['format_example']
                        extracted_val = f"{number_part} {unit_part}".strip() if has_space else f"{number_part}{unit_part}".strip()

                elif rule['type'] == 'numeric_vpe':
                    number = match.group(1)
                    raw_unit = match.group(2).title() 
                    unit = rule['unit_mapping'].get(raw_unit.lower(), raw_unit)
                    extracted_val = f"{number} {unit}"
                        
                elif rule['type'] == 'numeric_thread':
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
                            
        predicted_values.append(extracted_val)
        
    print(f"5. Saving {args.split} Submission...")
    target_df['feature_value'] = predicted_values
    
    output_filename = f'submission_{args.split}.parquet'
    final_submission = target_df[['uid', 'feature_name', 'feature_value', 'feature_type']]
    final_submission.to_parquet(output_filename, index=False)
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()