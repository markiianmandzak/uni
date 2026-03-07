"""
Build Taxonomy Lookup Table with Frequency Counts
Creates a fast lookup dictionary for categorical and numeric features
"""

import pandas as pd
import pickle
import ast
from collections import Counter
from typing import Dict, List, Any


def parse_aggregated_values(aggregated_values: str) -> List[str]:
    """
    Parse aggregated_feature_values string into clean list of values.
    Input: "{'[Stahl]','[Edelstahl (A2)]'}"
    Output: ['Stahl', 'Edelstahl (A2)']
    """
    if pd.isna(aggregated_values) or not aggregated_values:
        return []
    
    try:
        # Parse the string as a Python set
        parsed_set = ast.literal_eval(aggregated_values)
        
        # Strip [ and ] from each value
        clean_values = []
        for val in parsed_set:
            # Remove surrounding brackets
            clean_val = val.strip('[]')
            clean_values.append(clean_val)
        
        return clean_values
    except:
        return []


def build_lookup_table(
    taxonomy_path: str = 'taxonomy/taxonomy.parquet',
    train_features_path: str = 'train/product_features.parquet',
    output_path: str = 'lookup.pkl'
):
    """Build and save the taxonomy lookup table with frequency counts"""
    
    print("Loading taxonomy...")
    taxonomy = pd.read_parquet(taxonomy_path)
    
    print(f"Building lookup table from {len(taxonomy)} taxonomy entries...")
    lookup = {}
    
    # Build initial lookup structure
    for _, row in taxonomy.iterrows():
        category = row['category']
        feature_name = row['feature_name']
        feature_type = row['feature_type']
        aggregated_values = row['aggregated_feature_values']
        
        # Parse the values
        values = parse_aggregated_values(aggregated_values)
        
        # Create lookup entry
        key = (category, feature_name)
        
        if feature_type == 'categorical':
            # Initialize Counter with zero counts for all known values
            lookup[key] = {
                'type': 'categorical',
                'values': Counter({v: 0 for v in values})
            }
        elif feature_type == 'numeric':
            # Store example strings
            lookup[key] = {
                'type': 'numeric',
                'examples': values
            }
    
    print(f"✓ Built lookup structure with {len(lookup)} entries")
    
    # Enrich categorical features with actual frequencies from training data
    print("\nLoading training features to compute frequencies...")
    train_features = pd.read_parquet(train_features_path)
    
    # Filter to categorical only
    categorical_features = train_features[train_features['feature_type'] == 'categorical']
    print(f"Processing {len(categorical_features)} categorical feature instances...")
    
    # Load products to get category mapping
    print("Loading training products for category mapping...")
    train_products = pd.read_parquet('train/products.parquet')
    uid_to_category = dict(zip(train_products['uid'], train_products['category']))
    
    # Count frequencies
    enriched_count = 0
    ignored_count = 0
    
    for _, row in categorical_features.iterrows():
        uid = row['uid']
        feature_name = row['feature_name']
        feature_value = row['feature_value']
        
        # Get category from uid
        category = uid_to_category.get(uid)
        if not category:
            ignored_count += 1
            continue
        
        key = (category, feature_name)
        
        # Only increment if key exists and value is in the Counter
        if key in lookup and feature_value in lookup[key]['values']:
            lookup[key]['values'][feature_value] += 1
            enriched_count += 1
        else:
            ignored_count += 1
    
    print(f"✓ Enriched {enriched_count} categorical values")
    print(f"  Ignored {ignored_count} unknown category/feature/value combinations")
    
    # Save lookup table
    print(f"\nSaving lookup table to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(lookup, f)
    
    print(f"✓ Saved lookup table")
    
    # Display statistics
    print("\n" + "="*80)
    print("LOOKUP TABLE STATISTICS")
    print("="*80)
    categorical_count = sum(1 for v in lookup.values() if v['type'] == 'categorical')
    numeric_count = sum(1 for v in lookup.values() if v['type'] == 'numeric')
    print(f"Total entries: {len(lookup)}")
    print(f"Categorical features: {categorical_count}")
    print(f"Numeric features: {numeric_count}")
    
    # Show sample access patterns
    print("\n" + "="*80)
    print("SAMPLE ACCESS PATTERNS")
    print("="*80)
    
    # Find a categorical example
    for key, value in lookup.items():
        if value['type'] == 'categorical' and len(value['values']) > 0:
            category, feature_name = key
            most_common = value['values'].most_common(1)
            if most_common and most_common[0][1] > 0:
                most_freq_val, freq = most_common[0]
                print(f"\nCategorical example:")
                print(f'  lookup[("{category}", "{feature_name}")]["values"].most_common(1)[0][0]')
                print(f'  → "{most_freq_val}" (frequency: {freq})')
                
                # Show specific value lookup
                sample_val = list(value['values'].keys())[0]
                sample_freq = value['values'][sample_val]
                print(f'  lookup[("{category}", "{feature_name}")]["values"]["{sample_val}"]')
                print(f'  → {sample_freq}')
                break
    
    # Find a numeric example
    for key, value in lookup.items():
        if value['type'] == 'numeric' and len(value['examples']) > 0:
            category, feature_name = key
            examples = value['examples'][:3]
            print(f"\nNumeric example:")
            print(f'  lookup[("{category}", "{feature_name}")]["examples"]')
            print(f'  → {examples}')
            break
    
    print("\n" + "="*80)
    
    return lookup


def main():
    """Main execution"""
    lookup = build_lookup_table()


if __name__ == '__main__':
    main()
