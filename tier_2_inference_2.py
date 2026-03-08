import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import difflib
import re

# --- Configuration ---
DATA_ROOT = "data/feature-normalization-hackathon/data"
MODEL_PATH = "./tier2_qa_model/final"
SUBMISSION_INPUT = "submission_val.parquet"
OUTPUT_FILE = "submission_val_TIER2_SNAPPED.parquet"
BATCH_SIZE = 128  # M3 Max sweet spot
MAX_LEN = 256

class QADataset(Dataset):
    def __init__(self, questions, contexts):
        self.questions = questions
        self.contexts = contexts
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, idx):
        return {"question": self.questions[idx], "context": self.contexts[idx]}

def clean_ml_string(s):
    if not isinstance(s, str): return ""
    s = s.replace("##", "")
    s = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', s)
    s = re.sub(r'(\d+)\s+°', r'\1°', s)
    s = s.replace(" - ", "-")
    return s.strip()

def main():
    print("1. Loading Models, Data, and Taxonomy...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    device = torch.device("mps")
    model.to(device).eval() 

    tax_df = pd.read_parquet(os.path.join(DATA_ROOT, 'taxonomy', 'taxonomy.parquet'))
    valid_values_dict = {}
    for _, row in tax_df.iterrows():
        extracted_vals = re.findall(r"\[(.*?)\]", str(row['aggregated_feature_values']))
        if extracted_vals:
            valid_values_dict[(row['category'], row['feature_name'])] = extracted_vals

    sub_df = pd.read_parquet(SUBMISSION_INPUT)
    products = pd.read_parquet(os.path.join(DATA_ROOT, "val/products.parquet"))
    merged = sub_df.merge(products[['uid', 'category', 'title', 'description']], on='uid', how='left')
    merged['context'] = (merged['title'].fillna('') + " " + merged['description'].fillna('')).astype(str)
    
    to_fill_mask = merged['feature_value'].isna()
    work_df = merged[to_fill_mask].copy()
    
    # ---> 10% SAMPLE FOR SPEED <---
    # Comment this line out to run on the entire dataset!
    work_df = work_df.iloc[::10].copy() 
    
    dataset = QADataset(
        questions=[f"Was ist der Wert für {r.feature_name}?" for r in work_df.itertuples()],
        contexts=work_df['context'].tolist()
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    print(f"2. Running ML Inference on {len(work_df)} rows...")
    raw_results = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = tokenizer(batch['question'], batch['context'], add_special_tokens=True,
                               return_tensors="pt", max_length=MAX_LEN, padding=True, truncation="only_second").to(device)
            outputs = model(**inputs)
            starts = torch.argmax(outputs.start_logits, dim=-1).cpu()
            ends = torch.argmax(outputs.end_logits, dim=-1).cpu()
            input_ids = inputs.input_ids.cpu()

            for j in range(len(starts)):
                if starts[j] <= 0:
                    raw_results.append(None)
                else:
                    ans_tokens = input_ids[j][starts[j] : ends[j] + 1]
                    ans = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()
                    raw_results.append(ans if (ans and len(ans) < 50) else None)

    print("3. Snapping ML predictions to exact Taxonomy matches...")
    work_df['raw_ml_pred'] = raw_results
    snapped_values = []
    
    for row in work_df.itertuples():
        raw_pred = row.raw_ml_pred
        if pd.isna(raw_pred):
            snapped_values.append(None)
            continue
            
        clean_pred = clean_ml_string(raw_pred)
        valid_options = valid_values_dict.get((row.category, row.feature_name), [])
        
        # 1. Exact Match
        exact_match = next((v for v in valid_options if v.lower() == clean_pred.lower()), None)
        if exact_match:
            snapped_values.append(exact_match)
            continue
            
        # 2. Number missing a Unit
        if any(char.isdigit() for char in clean_pred):
            num_match = next((v for v in valid_options if v.startswith(clean_pred) and len(v) > len(clean_pred)), None)
            if num_match:
                snapped_values.append(num_match)
                continue
                
        # 3. Fuzzy Match (Fallback)
        close_matches = difflib.get_close_matches(clean_pred, valid_options, n=1, cutoff=0.6)
        if close_matches:
            snapped_values.append(close_matches[0])
        else:
            snapped_values.append(None)

    work_df['feature_value'] = snapped_values
    
    print("4. Evaluating Sample Accuracy...")
    val_gt = pd.read_parquet(os.path.join(DATA_ROOT, "val/product_features.parquet"))
    eval_sample = work_df.merge(val_gt[['uid', 'feature_name', 'feature_value']], 
                                on=['uid', 'feature_name'], 
                                suffixes=('_pred', '_gt'))
    
    eval_sample['is_correct'] = eval_sample['feature_value_pred'] == eval_sample['feature_value_gt']
    pred_made = eval_sample[~eval_sample['feature_value_pred'].isna()]
    acc = pred_made['is_correct'].mean() if len(pred_made) > 0 else 0
    
    print("\n" + "="*50)
    print(f"📊 ML + SNAPPER RESULTS")
    print("="*50)
    print(f"Predictions Attempted: {len(pred_made)} / {len(eval_sample)}")
    print(f"Sample Accuracy:       {acc:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()