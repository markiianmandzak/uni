import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# --- Configuration ---
DATA_ROOT = "data/feature-normalization-hackathon/data"
MODEL_PATH = "./tier2_qa_model/final"
SUBMISSION_INPUT = "submission_val.parquet"
OUTPUT_FILE = "submission_final_val_SAMPLE.parquet"
BATCH_SIZE = 128  # Slightly lower for more stability on M3 Max
MAX_LEN = 256

class QADataset(Dataset):
    def __init__(self, questions, contexts):
        self.questions = questions
        self.contexts = contexts
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, idx):
        return {"question": self.questions[idx], "context": self.contexts[idx]}

def main():
    print("1. Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    
    device = torch.device("mps")
    model.to(device).eval() 

    print("2. Preparing Data...")
    sub_df = pd.read_parquet(SUBMISSION_INPUT)
    products = pd.read_parquet(os.path.join(DATA_ROOT, "val/products.parquet"))
    merged = sub_df.merge(products[['uid', 'category', 'title', 'description']], on='uid', how='left')
    merged['context'] = (merged['title'].fillna('') + " " + merged['description'].fillna('')).astype(str)
    
    to_fill_mask = merged['feature_value'].isna()
    full_work_df = merged[to_fill_mask].copy()
    
    # SELECT EVERY 10th ROW
    work_df = full_work_df.iloc[::10].copy()
    
    dataset = QADataset(
        questions=[f"Was ist der Wert für {r.feature_name}?" for r in work_df.itertuples()],
        contexts=work_df['context'].tolist()
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0) # workers=0 avoids multiprocessing overhead for sample

    print(f"3. Running Inference on {len(work_df)} rows (1/10th sample)...")
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = tokenizer(
                batch['question'],
                batch['context'],
                add_special_tokens=True,
                return_tensors="pt",
                max_length=MAX_LEN,
                padding=True,
                truncation="only_second"
            ).to(device)

            outputs = model(**inputs)
            
            starts = torch.argmax(outputs.start_logits, dim=-1).cpu()
            ends = torch.argmax(outputs.end_logits, dim=-1).cpu()
            input_ids = inputs.input_ids.cpu()

            for j in range(len(starts)):
                if starts[j] <= 0:
                    results.append(None)
                else:
                    ans_tokens = input_ids[j][starts[j] : ends[j] + 1]
                    ans = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()
                    results.append(ans if (ans and len(ans) < 50) else None)

    # 4. Finalizing Sample
    work_df['feature_value'] = results
    
    # To evaluate properly, we need the Ground Truth for these specific rows
    val_gt = pd.read_parquet(os.path.join(DATA_ROOT, "val/product_features.parquet"))
    eval_sample = work_df.merge(val_gt[['uid', 'feature_name', 'feature_value']], 
                                on=['uid', 'feature_name'], 
                                suffixes=('_pred', '_gt'))
    
    eval_sample['is_correct'] = eval_sample['feature_value_pred'] == eval_sample['feature_value_gt']
    
    pred_made = eval_sample[~eval_sample['feature_value_pred'].isna()]
    acc = pred_made['is_correct'].mean() if len(pred_made) > 0 else 0
    
    print("\n" + "="*50)
    print(f"📊 SAMPLE RESULTS (88k rows sampled)")
    print("="*50)
    print(f"Predictions Attempted: {len(pred_made)} / {len(eval_sample)}")
    print(f"Sample Accuracy:       {acc:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()