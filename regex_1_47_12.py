import pandas as pd
import re
import argparse
from tqdm import tqdm
import os

# The actual root of your parquet files based on your tree
DATA_ROOT = "data/feature-normalization-hackathon/data"

def parse_taxonomy(tax_df):
    print("1. Parsing Taxonomy...")
    taxonomy_rules = {}
    
    for _, row in tax_df.iterrows():
        cat = row['category']
        feat = row['feature_name']
        f_type = row['feature_type']
        raw_vals = str(row['aggregated_feature_values'])
        
        extracted_vals = re.findall(r"\[(.*?)\]", raw_vals)
        
        if f_type == 'categorical':
            valid_strings = sorted(extracted_vals, key=len, reverse=True)
            if not valid_strings:
                continue
            # Pre-compile regex
            pattern_str = r'\b(' + '|'.join(map(re.escape, valid_strings)) + r')\b'
            taxonomy_rules[(cat, feat)] = {
                'type': 'categorical',
                'values': valid_strings,
                'pattern': re.compile(pattern_str, re.IGNORECASE)
            }
            
        elif f_type == 'numeric':
            if not extracted_vals:
                continue
            
            first_val = extracted_vals[0]
            parts = first_val.split()
            unit = parts[-1] if len(parts) > 1 else ""
            
            # Simple numeric regex
            regex_str = r"(\d+(?:[.,]\d+)?)\s*(" + re.escape(unit) + r")"
            taxonomy_rules[(cat, feat)] = {
                'type': 'numeric',
                'unit': unit,
                'pattern': re.compile(regex_str, re.IGNORECASE)
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
    
    # 2. Load Products and Target Template based on split
    print(f"2. Loading {args.split.upper()} Products and Target Template...")
    prod_path = os.path.join(DATA_ROOT, args.split, 'products.parquet')
    prod_df = pd.read_parquet(prod_path)
    
    if args.split == 'test':
        target_path = os.path.join(DATA_ROOT, 'test', 'submission.parquet')
        target_df = pd.read_parquet(target_path)
    else:
        # For validation, we use the ground truth file as our template but ignore the true values
        target_path = os.path.join(DATA_ROOT, 'val', 'product_features.parquet')
        target_df = pd.read_parquet(target_path)
        target_df['feature_value'] = None # Clear true values so we don't cheat!
    
    print("3. Merging Text into Template...")
    # Merge on uid to get the title and description for each required prediction
    merged_df = target_df.merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    merged_df['text'] = (merged_df['title'].fillna('') + " " + merged_df['description'].fillna('')).astype(str)
    
    print("4. Running Extraction...")
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
                    extracted_val = f"{number_part} {unit_part}".strip()
                    
        predicted_values.append(extracted_val)
        
    print(f"5. Saving Submission for {args.split}...")
    target_df['feature_value'] = predicted_values
    
    output_filename = f'submission_{args.split}.parquet'
    final_submission = target_df[['uid', 'feature_name', 'feature_value', 'feature_type']]
    final_submission.to_parquet(output_filename, index=False)
    print(f"Done! Saved to {output_filename}")

if __name__ == "__main__":
    main()