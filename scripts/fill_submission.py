# -*- coding: utf-8 -*-
"""
Fill test submission using existing extractor pipelines.
Imports predict functions from numeric_extractor.py and category_extractor.py.
"""
import pandas as pd
import numpy as np
import time
import sys
import os

print("=" * 80)
print("FILL TEST SUBMISSION")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Import extractors (triggers data loading + matcher compilation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Importing numeric extractor...")
t0 = time.time()
import numeric_extractor
print(f"  -> numeric extractor loaded in {time.time()-t0:.1f}s")
print(f"     {len(numeric_extractor.matchers)} matchers, {len(numeric_extractor.tax_values)} taxonomy pairs")

print("\n[2/5] Importing categorical extractor...")
t1 = time.time()
import category_extractor
print(f"  -> categorical extractor loaded in {time.time()-t1:.1f}s")
print(f"     {len(category_extractor.matchers)} matchers, {len(category_extractor.tax_values)} taxonomy pairs")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load test data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Loading test data...")
t2 = time.time()

submission = pd.read_parquet("data/test/submission.parquet")
products_test = pd.read_parquet("data/test/products.parquet")

print(f"  Submission rows: {len(submission):,}")
print(f"  Test products:   {len(products_test):,}")
print(f"  Feature types:   {submission['feature_type'].value_counts().to_dict()}")
print(f"  Loaded in {time.time()-t2:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build test text cache
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Building text cache for test products...")
t3 = time.time()

uid_text_cache = {}
for _, row in products_test.iterrows():
    uid = row['uid']
    title = row.get('title', '') or ''
    desc = row.get('description', '') or ''
    title_clean = numeric_extractor.clean_text(title).lower()
    desc_clean = numeric_extractor.clean_text(desc).lower()
    full_clean = title_clean + " " + desc_clean
    uid_text_cache[uid] = (title_clean, full_clean)

print(f"  Cached {len(uid_text_cache):,} products in {time.time()-t3:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Fill submission
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Filling submission...")
t4 = time.time()

# Merge to get category for each row
submission_merged = submission.merge(
    products_test[['uid', 'category']],
    on='uid', how='left'
)

total = len(submission_merged)
filled = 0
num_filled = 0
cat_filled = 0
results = []

for i, (_, row) in enumerate(submission_merged.iterrows()):
    uid = row['uid']
    category = row['category']
    feature_name = row['feature_name']
    feature_type = row['feature_type']

    text_data = uid_text_cache.get(uid)
    if text_data:
        title_lower, full_lower = text_data
    else:
        title_lower = ''
        full_lower = ''

    predicted = None

    if feature_type == 'numeric':
        predicted = numeric_extractor.predict_numeric_row(
            title_lower, full_lower, feature_name, category
        )
        if predicted:
            num_filled += 1
    else:
        predicted = category_extractor.predict_categorical_row(
            title_lower, full_lower, feature_name, category
        )
        if predicted:
            cat_filled += 1

    results.append(predicted)

    if predicted:
        filled += 1

    if (i + 1) % 500000 == 0:
        elapsed = time.time() - t4
        rate = (i + 1) / elapsed
        pct = 100 * filled / (i + 1)
        print(f"  [{i+1:>8,}/{total:,}] filled={pct:.1f}% "
              f"(num={num_filled:,} cat={cat_filled:,}) "
              f"rate={rate:,.0f}/s elapsed={elapsed:.0f}s")

elapsed = time.time() - t4
fill_pct = 100 * filled / total if total > 0 else 0

print(f"\n{'='*60}")
print(f"  FILLING COMPLETE")
print(f"  Total rows:     {total:,}")
print(f"  Filled:         {filled:,} ({fill_pct:.1f}%)")
print(f"  Numeric filled: {num_filled:,}")
print(f"  Cat filled:     {cat_filled:,}")
print(f"  Unfilled:       {total - filled:,}")
print(f"  Time:           {elapsed:.1f}s ({total/max(elapsed,0.1):,.0f} rows/sec)")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Save result
# ─────────────────────────────────────────────────────────────────────────────
print("\nSaving submission...")

# Copy original submission and fill values
output = submission.copy()
output['feature_value'] = results

# Ensure no NaN — keep None as is (or empty string)
output['feature_value'] = output['feature_value'].fillna('')

output_path = "data/test/submission_filled_01.parquet"
output.to_parquet(output_path, index=False)

print(f"Saved to {output_path}")
print(f"Shape: {output.shape}")
print(f"Non-empty values: {(output['feature_value'] != '').sum():,} / {len(output):,}")
print(f"\nTotal pipeline time: {time.time()-t0:.1f}s")
