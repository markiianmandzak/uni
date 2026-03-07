"""
Reshuffle and Resplit Train/Val Data
Merges train and val, shuffles, and splits 95/5
"""

import pandas as pd
import numpy as np
from pathlib import Path


def reshuffle_split(train_ratio: float = 0.95, random_seed: int = 42):
    """Merge, shuffle, and split train/val data"""
    
    print("="*80)
    print("RESHUFFLING DATA")
    print("="*80)
    
    # Load original data
    print("\nLoading original data...")
    train_products = pd.read_parquet('train/products.parquet')
    train_features = pd.read_parquet('train/product_features.parquet')
    val_products = pd.read_parquet('val/products.parquet')
    val_features = pd.read_parquet('val/product_features.parquet')
    
    print(f"Train products: {len(train_products)}")
    print(f"Train features: {len(train_features)}")
    print(f"Val products: {len(val_products)}")
    print(f"Val features: {len(val_features)}")
    
    # Merge train and val
    print("\nMerging train and val...")
    all_products = pd.concat([train_products, val_products], ignore_index=True)
    all_features = pd.concat([train_features, val_features], ignore_index=True)
    
    print(f"Total products: {len(all_products)}")
    print(f"Total features: {len(all_features)}")
    
    # Shuffle products
    print(f"\nShuffling with seed {random_seed}...")
    all_products = all_products.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    split_idx = int(len(all_products) * train_ratio)
    
    new_train_products = all_products[:split_idx]
    new_val_products = all_products[split_idx:]
    
    print(f"\nSplit at index {split_idx}:")
    print(f"New train products: {len(new_train_products)} ({len(new_train_products)/len(all_products)*100:.1f}%)")
    print(f"New val products: {len(new_val_products)} ({len(new_val_products)/len(all_products)*100:.1f}%)")
    
    # Get UIDs for split
    train_uids = set(new_train_products['uid'])
    val_uids = set(new_val_products['uid'])
    
    # Split features based on UIDs
    new_train_features = all_features[all_features['uid'].isin(train_uids)]
    new_val_features = all_features[all_features['uid'].isin(val_uids)]
    
    print(f"\nNew train features: {len(new_train_features)}")
    print(f"New val features: {len(new_val_features)}")
    
    # Create output directories
    print("\nCreating output directories...")
    Path('train_reshuffled').mkdir(exist_ok=True)
    Path('val_reshuffled').mkdir(exist_ok=True)
    
    # Save
    print("Saving reshuffled data...")
    new_train_products.to_parquet('train_reshuffled/products.parquet', index=False)
    new_train_features.to_parquet('train_reshuffled/product_features.parquet', index=False)
    new_val_products.to_parquet('val_reshuffled/products.parquet', index=False)
    new_val_features.to_parquet('val_reshuffled/product_features.parquet', index=False)
    
    print("✓ Saved to train_reshuffled/ and val_reshuffled/")


if __name__ == '__main__':
    reshuffle_split(train_ratio=0.95, random_seed=42)
