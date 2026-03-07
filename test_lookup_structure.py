import pickle

# Load lookup
with open('lookup.pkl', 'rb') as f:
    lookup = pickle.load(f)

print("="*80)
print("LOOKUP TABLE STRUCTURE TEST")
print("="*80)

print(f"\nTotal categories: {len(lookup)}")

# Show first category
first_cat = list(lookup.keys())[0]
print(f"\nFirst category: '{first_cat}'")
print(f"Features in this category: {len(lookup[first_cat])}")

# Show first feature
first_feat = list(lookup[first_cat].keys())[0]
print(f"\nFirst feature: '{first_feat}'")
print(f"Values: {lookup[first_cat][first_feat]}")

# Show a few more examples
print("\n" + "="*80)
print("SAMPLE LOOKUPS")
print("="*80)

count = 0
for category in lookup:
    if count >= 3:
        break
    for feature_name in list(lookup[category].keys())[:1]:
        values = lookup[category][feature_name]
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        print(f'\nlookup["{category}"]["{feature_name}"] = {{')
        for val, pct in sorted_values[:3]:
            print(f'  "{val}": {pct:.2f}%')
        if len(sorted_values) > 3:
            print(f'  ... ({len(sorted_values)} total values)')
        print('}')
        count += 1
        break

print("\n" + "="*80)
