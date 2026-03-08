import pandas as pd
import json
from tqdm import tqdm
import os

# Set your data root
DATA_ROOT = "data/feature-normalization-hackathon/data"

def create_qa_dataset(split='train'):
    print(f"1. Loading {split.upper()} data...")
    products = pd.read_parquet(os.path.join(DATA_ROOT, split, "products.parquet"))
    features = pd.read_parquet(os.path.join(DATA_ROOT, split, "product_features.parquet"))
    
    # Merge text and features
    df = features.merge(products[['uid', 'title', 'description']], on='uid', how='inner')
    
    # Combine title and description into one searchable context
    df['context'] = (df['title'].fillna('') + " \n " + df['description'].fillna('')).astype(str)
    
    dataset = []
    dropped_count = 0
    
    print("2. Finding exact answer spans in text...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        context = row.context
        ans = str(row.feature_value)
        feat_name = str(row.feature_name)
        
        # We only want to train on rows where the answer is actually written in the text
        if ans and ans != 'None':
            # Find where the answer starts
            start_idx = context.find(ans)
            
            if start_idx != -1:
                dataset.append({
                    "id": f"{row.uid}_{feat_name}",
                    "context": context,
                    "question": f"Was ist der Wert für {feat_name}?",
                    "answers": {
                        "text": [ans],
                        "answer_start": [start_idx]
                    }
                })
            else:
                # Answer is implied or formatted differently (we skip these for simple QA training)
                dropped_count += 1
                
    output_file = f"{split}_qa_dataset.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Success!")
    print(f"Saved {len(dataset)} perfect training examples to {output_file}")
    print(f"Dropped {dropped_count} examples where the exact string wasn't found in the text.")

if __name__ == "__main__":
    create_qa_dataset('train')
    # Optional: You can also run it for 'val' to create a validation set for your ML model
    create_qa_dataset('val')
    