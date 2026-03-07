"""
Build Lookup Table with Frequency Percentages
Structure: lookup[category][feature_name] = {value: percentage, ...}
"""

import pandas as pd
import pickle
from collections import defaultdict, Counter


def build_lookup_table(
    train_products_path: str = 'train/products.parquet',
    train_features_path: str = 'train/product_features.parquet',
    output_path: str = 'lookup.pkl'
):
    """Build lookup table with frequency percentages for categorical features"""
    
    print("Loading data...")
    train_products = pd.read_parquet(train_products_path)
    train_features = pd.read_parquet(train_features_path)
    
    print("Merging products and features on uid...")
    merged = train_features.merge(train_products[['uid', 'category']], on='uid', how='left')
    
    # Filter to categorical only
    categorical = merged[merged['feature_type'] == 'categorical']
    print(f"Processing {len(categorical)} categorical instances...")
    
    # Count frequencies: category -> feature_name -> value -> count
    freq_counts = defaultdict(lambda: defaultdict(Counter))
    
    for _, row in categorical.iterrows():
        category = row['category']
        feature_name = row['feature_name']
        feature_value = row['feature_value']
        freq_counts[category][feature_name][feature_value] += 1
    
    # Convert counts to percentages
    print("Converting counts to percentages...")
    lookup = {}
    
    for category in freq_counts:
        lookup[category] = {}
        for feature_name in freq_counts[category]:
            counter = freq_counts[category][feature_name]
            total = sum(counter.values())
            
            # Calculate percentages
            lookup[category][feature_name] = {
                val: (count / total) * 100 
                for val, count in counter.items()
            }
    
    # Save lookup table
    print(f"\nSaving lookup table to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(lookup, f)
    
    print(f"✓ Saved lookup table")
    
    return lookup


if __name__ == '__main__':
    build_lookup_table()
