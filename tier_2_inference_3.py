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
SUBMISSION_INPUT = "submission_val.parquet" # Or submission_test.parquet
OUTPUT_FILE = "submission_final_merged.parquet"
BATCH_SIZE = 256  # Optimized for M3 Max 40-core GPU
MAX_LEN = 256
NUM_WORKERS = 8   # parallel tokenization on CPU

# --- Unit Conversion Logic ---
CONVERSIONS = {
    'mm': {'cm': 0.1, 'm': 0.001, 'inch': 0.03937},
    'cm': {'mm': 10, 'm': 0.01},
    'm': {'mm': 1000, 'cm': 100},
}

class QADataset(Dataset):
    def __init__(self, questions, contexts):
        self.questions = questions
        self.contexts = contexts
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, idx):
        return {"question": self.questions[idx], "context": self.contexts[idx]}

def normalize_ml_numeric(pred, valid_options):
    """
    Tries to convert ML extraction (e.g. '15 cm') to a taxonomy valid value (e.g. '150 mm').
    """
    pred_clean = pred.lower().replace(',', '.')
    match = re.search(r'([\d.]+)\s*([a-z]+)?', pred_clean)
    if not match or not valid_options: return None
    
    val = float(match.group(1))
    unit = match.group(2)
    
    for opt in valid_options:
        opt_match = re.search(r'([\d.]+)\s*([a-z]+)?', opt.lower().replace(',', '.'))
        if opt_match:
            target_val = float(opt_match.group(1))
            target_unit = opt_match.group(2)
            
            # Simple check: same value, same unit
            if unit == target_unit and val == target_val: return opt
            
            # Unit conversion check
            if unit in CONVERSIONS and target_unit in CONVERSIONS[unit]:
                if abs((val * CONVERSIONS[unit][target_unit]) - target_val) < 1e-5:
                    return opt
    return None

def clean_ml_string(s):
    if not isinstance(s, str): return ""
    s = s.replace("##", "")
    s = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', s)
    s = s.replace(" - ", "-")
    return s.strip()

def main():
    device = torch.device("mps")
    print(f"1. Loading Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    model.to(device).eval()

    print("2. Preparing Data...")
    tax_df = pd.read_parquet(os.path.join(DATA_ROOT, 'taxonomy', 'taxonomy.parquet'))
    valid_values_dict = {}
    for _, row in tax_df.iterrows():
        vals = re.findall(r"\[(.*?)\]", str(row['aggregated_feature_values']))
        if vals: valid_values_dict[(row['category'], row['feature_name'])] = vals

    # Load Tier 1 results
    sub_df = pd.read_parquet(SUBMISSION_INPUT)
    split = "test" if "test" in SUBMISSION_INPUT else "val"
    products = pd.read_parquet(os.path.join(DATA_ROOT, f"{split}/products.parquet"))
    
    merged = sub_df.merge(products[['uid', 'category', 'title', 'description']], on='uid', how='left')
    merged['context'] = (merged['title'].fillna('') + " " + merged['description'].fillna('')).astype(str)
    
    # Only process what Tier 1 missed
    to_fill_mask = merged['feature_value'].isna()
    work_df = merged[to_fill_mask].copy()
    
    # SPEED HACK: Sort by length to minimize padding in batches
    work_df['ctx_len'] = work_df['context'].str.len()
    work_df = work_df.sort_values('ctx_len')

    dataset = QADataset(
        questions=[f"Was ist {r.feature_name}?" for r in work_df.itertuples()],
        contexts=work_df['context'].tolist()
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    print(f"3. Inference on {len(work_df)} missing values...")
    raw_results = []
    with torch.inference_mode(): # Fastest inference on MPS
        for batch in tqdm(loader):
            inputs = tokenizer(batch['question'], batch['context'], add_special_tokens=True,
                               return_tensors="pt", max_length=MAX_LEN, padding=True, truncation="only_second").to(device)
            outputs = model(**inputs)
            
            starts = torch.argmax(outputs.start_logits, dim=-1).cpu()
            ends = torch.argmax(outputs.end_logits, dim=-1).cpu()
            ids = inputs.input_ids.cpu()

            for j in range(len(starts)):
                if starts[j] <= 0 or ends[j] < starts[j]:
                    raw_results.append(None)
                else:
                    ans = tokenizer.decode(ids[j][starts[j] : ends[j] + 1], skip_special_tokens=True).strip()
                    raw_results.append(ans if (ans and len(ans) < 60) else None)

    print("4. Snapping & Normalizing...")
    work_df['ml_raw'] = raw_results
    final_tier2_preds = []
    
    for row in work_df.itertuples():
        raw = row.ml_raw
        if not raw:
            final_tier2_preds.append(None)
            continue
            
        clean = clean_ml_string(raw)
        options = valid_values_dict.get((row.category, row.feature_name), [])
        
        # Priority 1: Exact Match
        match = next((v for v in options if v.lower() == clean.lower()), None)
        
        # Priority 2: Numeric Conversion (cm -> mm, etc)
        if not match and any(c.isdigit() for c in clean):
            match = normalize_ml_numeric(clean, options)
            
        # Priority 3: Fuzzy
        if not match:
            close = difflib.get_close_matches(clean, options, n=1, cutoff=0.7)
            match = close[0] if close else None
            
        final_tier2_preds.append(match)

    work_df['feature_value'] = final_tier2_preds
    
    print("5. Merging Tier 1 + Tier 2...")
    # Map the new predictions back to the original dataframe
    prediction_map = dict(zip(zip(work_df.uid, work_df.feature_name), work_df.feature_value))
    
    def fill_val(row):
        if pd.isna(row.feature_value):
            return prediction_map.get((row.uid, row.feature_name), None)
        return row.feature_value

    sub_df['feature_value'] = sub_df.apply(fill_val, axis=1)
    
    sub_df[['uid', 'feature_name', 'feature_value', 'feature_type']].to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Success! Final submission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()