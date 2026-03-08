import pandas as pd
import re
import argparse
from tqdm import tqdm
import os

# Set this to your actual data path
DATA_ROOT = "data/feature-normalization-hackathon/data"

def parse_taxonomy(tax_df):
    print("1. Building Categorical-Only Ruleset...")
    taxonomy_rules = {}

    for _, row in tax_df.iterrows():
        cat = row['category']
        feat = row['feature_name']
        f_type = row['feature_type']
        raw_vals = str(row['aggregated_feature_values'])
        
        # We completely ignore anything that isn't categorical
        if f_type != 'categorical':
            continue
            
        extracted_vals = re.findall(r"\[(.*?)\]", raw_vals)
        if not extracted_vals:
            continue
            
        # Sort by length descending to match "Edelstahl (A2)" before "Stahl"
        valid_strings = sorted(extracted_vals, key=len, reverse=True)
        
        # Using strict word boundaries to prevent matching "Stahl" inside "Edelstahl"
        pattern_str = r'\b(' + '|'.join(map(re.escape, valid_strings)) + r')\b'
        
        taxonomy_rules[(cat, feat)] = {
            'type': 'categorical',
            'values': valid_strings,
            'pattern': re.compile(pattern_str, re.IGNORECASE)
        }
                
    return taxonomy_rules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val')
    args = parser.parse_args()

    # 1. Load Taxonomy
    tax_path = os.path.join(DATA_ROOT, 'taxonomy', 'taxonomy.parquet')
    tax_df = pd.read_parquet(tax_path)
    taxonomy_rules = parse_taxonomy(tax_df)
    
    # 2. Load Products and Target Template
    print(f"2. Loading {args.split.upper()} Products and Target Template...")
    prod_path = os.path.join(DATA_ROOT, args.split, 'products.parquet')
    prod_df = pd.read_parquet(prod_path)
    
    if args.split == 'test':
        target_path = os.path.join(DATA_ROOT, 'test', 'submission.parquet')
        target_df = pd.read_parquet(target_path)
    else:
        target_path = os.path.join(DATA_ROOT, 'val', 'product_features.parquet')
        target_df = pd.read_parquet(target_path)
        target_df['feature_value'] = None 
    
    print("3. Merging Text into Template...")
    merged_df = target_df.merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    merged_df['text'] = (merged_df['title'].fillna('') + " " + merged_df['description'].fillna('')).astype(str)
    
    print("4. Running Categorical Extraction...")
    predicted_values = []
    
    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        cat = row.category
        feat = row.feature_name
        text = row.text
        
        rule = taxonomy_rules.get((cat, feat))
        extracted_val = None
        
        # If the rule exists, it's a categorical feature. Otherwise, it stays None.
        if rule and text.strip():
            match = rule['pattern'].search(text)
            if match:
                matched_str_lower = match.group(1).lower()
                # Map back to the exact casing required by the taxonomy
                for v in rule['values']:
                    if v.lower() == matched_str_lower:
                        extracted_val = v
                        break
                            
        predicted_values.append(extracted_val)
        
    print(f"5. Saving Submission for {args.split}...")
    target_df['feature_value'] = predicted_values
    
    output_filename = f'submission_{args.split}.parquet'
    final_submission = target_df[['uid', 'feature_name', 'feature_value', 'feature_type']]
    final_submission.to_parquet(output_filename, index=False)
    print(f"Done! Saved to {output_filename}")

if __name__ == "__main__":
    main()
    