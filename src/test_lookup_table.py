"""
Test Lookup Table Performance
Evaluates how well the lookup table handles categorical and numeric features
"""

import pickle
import pandas as pd
from collections import Counter


def load_lookup(path: str = 'lookup.pkl'):
    """Load the lookup table"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def test_categorical_coverage(lookup, val_features_path: str = 'val/product_features.parquet'):
    """Test categorical feature coverage and frequency matching"""
    print("="*80)
    print("CATEGORICAL FEATURES TEST")
    print("="*80)
    
    # Load validation data
    val_features = pd.read_parquet(val_features_path)
    val_products = pd.read_parquet('val/products.parquet')
    uid_to_category = dict(zip(val_products['uid'], val_products['category']))
    
    # Filter categorical
    categorical = val_features[val_features['feature_type'] == 'categorical'].copy()
    categorical['category'] = categorical['uid'].map(uid_to_category)
    
    print(f"\nTotal categorical instances in validation: {len(categorical)}")
    
    # Test coverage
    found_in_lookup = 0
    value_matches = 0
    value_mismatches = 0
    missing_keys = 0
    
    for _, row in categorical.iterrows():
        category = row['category']
        feature_name = row['feature_name']
        feature_value = row['feature_value']
        
        key = (category, feature_name)
        
        if key not in lookup:
            missing_keys += 1
            continue
        
        found_in_lookup += 1
        
        # Check if value exists in lookup
        if feature_value in lookup[key]['values']:
            value_matches += 1
        else:
            value_mismatches += 1
    
    print(f"\nCoverage:")
    print(f"  Features found in lookup: {found_in_lookup}/{len(categorical)} ({100*found_in_lookup/len(categorical):.2f}%)")
    print(f"  Missing keys: {missing_keys}")
    
    print(f"\nValue Matching:")
    print(f"  Values in taxonomy: {value_matches}/{found_in_lookup} ({100*value_matches/found_in_lookup:.2f}%)")
    print(f"  Values NOT in taxonomy: {value_mismatches}/{found_in_lookup} ({100*value_mismatches/found_in_lookup:.2f}%)")
    
    # Test frequency-based prediction
    print(f"\n" + "-"*80)
    print("FREQUENCY-BASED PREDICTION TEST")
    print("-"*80)
    
    correct_predictions = 0
    total_predictions = 0
    
    for _, row in categorical.iterrows():
        category = row['category']
        feature_name = row['feature_name']
        feature_value = row['feature_value']
        
        key = (category, feature_name)
        
        if key not in lookup:
            continue
        
        # Predict most frequent value
        most_common = lookup[key]['values'].most_common(1)
        if most_common and most_common[0][1] > 0:
            predicted_value = most_common[0][0]
            total_predictions += 1
            
            if predicted_value == feature_value:
                correct_predictions += 1
    
    if total_predictions > 0:
        accuracy = 100 * correct_predictions / total_predictions
        print(f"\nMost-frequent baseline accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2f}%)")
    
    # Show some examples
    print(f"\n" + "-"*80)
    print("SAMPLE CATEGORICAL LOOKUPS")
    print("-"*80)
    
    sample_count = 0
    for _, row in categorical.head(100).iterrows():
        if sample_count >= 5:
            break
        
        category = row['category']
        feature_name = row['feature_name']
        feature_value = row['feature_value']
        
        key = (category, feature_name)
        
        if key in lookup and len(lookup[key]['values']) > 0:
            top_3 = lookup[key]['values'].most_common(3)
            print(f"\n{category} | {feature_name}")
            print(f"  Ground truth: {feature_value}")
            print(f"  Top 3 values: {top_3}")
            sample_count += 1


def test_numeric_coverage(lookup, val_features_path: str = 'val/product_features.parquet'):
    """Test numeric feature coverage"""
    print("\n" + "="*80)
    print("NUMERIC FEATURES TEST")
    print("="*80)
    
    # Load validation data
    val_features = pd.read_parquet(val_features_path)
    val_products = pd.read_parquet('val/products.parquet')
    uid_to_category = dict(zip(val_products['uid'], val_products['category']))
    
    # Filter numeric
    numeric = val_features[val_features['feature_type'] == 'numeric'].copy()
    numeric['category'] = numeric['uid'].map(uid_to_category)
    
    print(f"\nTotal numeric instances in validation: {len(numeric)}")
    
    # Test coverage
    found_in_lookup = 0
    missing_keys = 0
    
    for _, row in numeric.iterrows():
        category = row['category']
        feature_name = row['feature_name']
        
        key = (category, feature_name)
        
        if key not in lookup:
            missing_keys += 1
        else:
            found_in_lookup += 1
    
    print(f"\nCoverage:")
    print(f"  Features found in lookup: {found_in_lookup}/{len(numeric)} ({100*found_in_lookup/len(numeric):.2f}%)")
    print(f"  Missing keys: {missing_keys}")
    
    # Show some examples
    print(f"\n" + "-"*80)
    print("SAMPLE NUMERIC LOOKUPS")
    print("-"*80)
    
    sample_count = 0
    for _, row in numeric.head(100).iterrows():
        if sample_count >= 5:
            break
        
        category = row['category']
        feature_name = row['feature_name']
        feature_value = row['feature_value']
        
        key = (category, feature_name)
        
        if key in lookup and len(lookup[key]['examples']) > 0:
            examples = lookup[key]['examples'][:5]
            print(f"\n{category} | {feature_name}")
            print(f"  Ground truth: {feature_value}")
            print(f"  Example values: {examples}")
            sample_count += 1


def main():
    """Main execution"""
    print("\nLoading lookup table...")
    lookup = load_lookup()
    print(f"✓ Loaded {len(lookup)} entries\n")
    
    # Test categorical
    test_categorical_coverage(lookup)
    
    # Test numeric
    test_numeric_coverage(lookup)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
