import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

import regex_14 as r
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings('ignore', category=ConvergenceWarning)

TARGET_FEATURES = [
    'Ausführung', 'Gewinde-Ø', 'Material', 'Schneidstoff', 'Oberfläche',
    'Antriebsgröße', 'Formfaktor', 'Antrieb', 'Form'
]
TRANSFORMER_FEATURES = ['Schneidstoff', 'Antriebsgröße', 'Formfaktor', 'Antrieb']
LOCAL_MODEL_PATH = 'tier2_qa_model/final'


def no_tqdm(iterable, total=None):
    return iterable


def normalize_text(value):
    return r.normalize_text(value)


def row_text(category, feature_name, title, description):
    title_norm = normalize_text(title)
    desc_norm = normalize_text(description)
    return f"[CAT] {category} [FEAT] {feature_name} [TITLE] {title_norm} [DESC] {desc_norm}".strip()


def sample_training_rows(train_df, max_rows=15000, max_per_label=400, min_count=2):
    pieces = []
    counts = train_df['feature_value'].value_counts()
    keep_labels = set(counts[counts >= min_count].index)
    train_df = train_df[train_df['feature_value'].isin(keep_labels)]
    for label, group in train_df.groupby('feature_value', sort=False):
        take = min(len(group), max_per_label)
        pieces.append(group.sample(take, random_state=42) if len(group) > take else group)
    if not pieces:
        return train_df.iloc[0:0].copy()
    sampled = pd.concat(pieces, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=42)
    return sampled.reset_index(drop=True)


def build_eval_report(sample_size):
    r.tqdm = no_tqdm
    tax_df = pd.read_parquet(os.path.join(r.DATA_ROOT, 'taxonomy', 'taxonomy.parquet'))
    taxonomy_rules = r.build_taxonomy_rules(tax_df)

    train_feat_df = pd.read_parquet(os.path.join(r.DATA_ROOT, 'train', 'product_features.parquet'), columns=['uid', 'feature_name', 'feature_value'])
    feature_modes = r.build_feature_modes(train_feat_df[['feature_name', 'feature_value']])
    train_prod_df = pd.read_parquet(os.path.join(r.DATA_ROOT, 'train', 'products.parquet'), columns=['uid', 'category', 'title', 'description'])
    train_joined_df = train_feat_df.merge(train_prod_df, on='uid', how='left')
    category_feature_modes = r.build_category_feature_modes(train_joined_df[['category', 'feature_name', 'feature_value']])
    mined_aliases = r.build_mined_categorical_aliases(
        train_joined_df[train_joined_df['feature_name'].isin(r.ALIAS_FEATURES)][['category', 'feature_name', 'feature_value', 'title']]
    )

    prod_df = pd.read_parquet(os.path.join(r.DATA_ROOT, 'val', 'products.parquet'))
    target_df = pd.read_parquet(os.path.join(r.DATA_ROOT, 'val', 'product_features.parquet')).sample(sample_size, random_state=42).copy()
    truth = target_df['feature_value'].copy()
    target_df['feature_value'] = None
    preds, stages = r.build_predictions(prod_df, target_df, taxonomy_rules, feature_modes, category_feature_modes, mined_aliases)

    report = target_df[['uid', 'feature_name', 'feature_type']].merge(prod_df[['uid', 'category', 'title', 'description']], on='uid', how='left')
    report['truth'] = truth.values
    report['pred'] = preds
    report['stage'] = stages
    report['correct'] = report['truth'] == report['pred']
    report['text'] = [row_text(c, f, t, d) for c, f, t, d in zip(report['category'], report['feature_name'], report['title'], report['description'])]
    report['allowed_values'] = [taxonomy_rules[(c, f)]['values'] if (c, f) in taxonomy_rules else [] for c, f in zip(report['category'], report['feature_name'])]
    return report, train_joined_df, taxonomy_rules


def argmax_allowed(prob_row, classes, allowed_values):
    allowed_positions = [idx for idx, value in enumerate(classes) if value in allowed_values]
    if not allowed_positions:
        return None, None, None
    allowed_probs = prob_row[allowed_positions]
    order = np.argsort(allowed_probs)[::-1]
    best_idx = allowed_positions[order[0]]
    best_value = classes[best_idx]
    best_prob = float(prob_row[best_idx])
    second_prob = float(allowed_probs[order[1]]) if len(order) > 1 else 0.0
    margin = best_prob - second_prob
    return best_value, best_prob, margin


def baseline_accuracy(eval_df):
    return float(eval_df['correct'].mean()) if len(eval_df) else 0.0


def evaluate_predictions(eval_df, candidate_preds, candidate_scores, candidate_margins, min_prob, min_margin):
    accepted = []
    final_preds = []
    for base_pred, cand_pred, score, margin in zip(eval_df['pred'], candidate_preds, candidate_scores, candidate_margins):
        use_model = cand_pred is not None and score is not None and margin is not None and score >= min_prob and margin >= min_margin
        accepted.append(use_model)
        final_preds.append(cand_pred if use_model else base_pred)
    final_preds = np.array(final_preds, dtype=object)
    accepted = np.array(accepted, dtype=bool)
    combined_acc = float((final_preds == eval_df['truth'].values).mean()) if len(eval_df) else 0.0
    accepted_acc = float((final_preds[accepted] == eval_df['truth'].values[accepted]).mean()) if accepted.any() else 0.0
    return {
        'accepted': int(accepted.sum()),
        'accepted_rate': float(accepted.mean()) if len(accepted) else 0.0,
        'accepted_acc': accepted_acc,
        'combined_acc': combined_acc,
    }


def benchmark_sklearn(eval_df, train_joined_df):
    from scipy.sparse import vstack
    all_candidate_preds = [None] * len(eval_df)
    all_candidate_scores = [None] * len(eval_df)
    all_candidate_margins = [None] * len(eval_df)
    feature_offsets = {}
    cursor = 0
    for feature_name, feature_rows in eval_df.groupby('feature_name', sort=False):
        feature_offsets[feature_name] = feature_rows.index.tolist()
        train_sub = train_joined_df[train_joined_df['feature_name'] == feature_name][['category', 'feature_name', 'title', 'description', 'feature_value']].copy()
        if len(train_sub) < 50 or train_sub['feature_value'].nunique() < 2:
            continue
        train_sub = sample_training_rows(train_sub)
        train_texts = [row_text(c, feature_name, t, d) for c, t, d in zip(train_sub['category'], train_sub['title'], train_sub['description'])]
        y = train_sub['feature_value'].values
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=60000, sublinear_tf=True)
        X_train = vectorizer.fit_transform(train_texts)
        clf = SGDClassifier(loss='log_loss', alpha=2e-6, penalty='l2', max_iter=25, tol=1e-3, random_state=42)
        clf.fit(X_train, y)
        eval_sub = eval_df.loc[feature_rows.index]
        X_eval = vectorizer.transform(eval_sub['text'])
        probs = clf.predict_proba(X_eval)
        classes = clf.classes_
        for row_idx, prob_row, allowed_values in zip(eval_sub.index, probs, eval_sub['allowed_values']):
            value, score, margin = argmax_allowed(prob_row, classes, allowed_values)
            pos = eval_df.index.get_loc(row_idx)
            all_candidate_preds[pos] = value
            all_candidate_scores[pos] = score
            all_candidate_margins[pos] = margin
    best = None
    for min_prob in (0.45, 0.55, 0.65, 0.75, 0.85):
        for min_margin in (0.00, 0.05, 0.10, 0.15, 0.20):
            stats = evaluate_predictions(eval_df, all_candidate_preds, all_candidate_scores, all_candidate_margins, min_prob, min_margin)
            stats['min_prob'] = min_prob
            stats['min_margin'] = min_margin
            if best is None or stats['combined_acc'] > best['combined_acc'] or (stats['combined_acc'] == best['combined_acc'] and stats['accepted_acc'] > best['accepted_acc']):
                best = stats
    best['method'] = 'sklearn_sgd_tfidf'
    return best


def benchmark_lgbm(eval_df, train_joined_df):
    import lightgbm as lgb
    all_candidate_preds = [None] * len(eval_df)
    all_candidate_scores = [None] * len(eval_df)
    all_candidate_margins = [None] * len(eval_df)
    for feature_name, feature_rows in eval_df.groupby('feature_name', sort=False):
        train_sub = train_joined_df[train_joined_df['feature_name'] == feature_name][['category', 'feature_name', 'title', 'description', 'feature_value']].copy()
        if len(train_sub) < 100 or train_sub['feature_value'].nunique() < 2:
            continue
        train_sub = sample_training_rows(train_sub, max_rows=12000, max_per_label=300)
        labels = sorted(train_sub['feature_value'].unique())
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        y = np.array([label_to_id[value] for value in train_sub['feature_value'].values])
        texts = [row_text(c, feature_name, t, d) for c, t, d in zip(train_sub['category'], train_sub['title'], train_sub['description'])]
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=40000, sublinear_tf=True)
        X_train = vectorizer.fit_transform(texts)
        svd_dim = max(16, min(128, X_train.shape[0] - 1, X_train.shape[1] - 1))
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        X_train_red = svd.fit_transform(X_train)
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(labels),
            n_estimators=160,
            learning_rate=0.08,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train_red, y)
        eval_sub = eval_df.loc[feature_rows.index]
        X_eval = svd.transform(vectorizer.transform(eval_sub['text']))
        probs = model.predict_proba(X_eval)
        for row_idx, prob_row, allowed_values in zip(eval_sub.index, probs, eval_sub['allowed_values']):
            value, score, margin = argmax_allowed(prob_row, labels, allowed_values)
            pos = eval_df.index.get_loc(row_idx)
            all_candidate_preds[pos] = value
            all_candidate_scores[pos] = score
            all_candidate_margins[pos] = margin
    best = None
    for min_prob in (0.35, 0.45, 0.55, 0.65, 0.75):
        for min_margin in (0.00, 0.05, 0.10, 0.15, 0.20):
            stats = evaluate_predictions(eval_df, all_candidate_preds, all_candidate_scores, all_candidate_margins, min_prob, min_margin)
            stats['min_prob'] = min_prob
            stats['min_margin'] = min_margin
            if best is None or stats['combined_acc'] > best['combined_acc'] or (stats['combined_acc'] == best['combined_acc'] and stats['accepted_acc'] > best['accepted_acc']):
                best = stats
    best['method'] = 'lightgbm_tfidf_svd'
    return best


def benchmark_catboost(eval_df, train_joined_df):
    from catboost import CatBoostClassifier
    all_candidate_preds = [None] * len(eval_df)
    all_candidate_scores = [None] * len(eval_df)
    all_candidate_margins = [None] * len(eval_df)
    for feature_name, feature_rows in eval_df.groupby('feature_name', sort=False):
        train_sub = train_joined_df[train_joined_df['feature_name'] == feature_name][['category', 'feature_name', 'title', 'description', 'feature_value']].copy()
        if len(train_sub) < 100 or train_sub['feature_value'].nunique() < 2:
            continue
        train_sub = sample_training_rows(train_sub, max_rows=10000, max_per_label=250)
        labels = sorted(train_sub['feature_value'].unique())
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        y = np.array([label_to_id[value] for value in train_sub['feature_value'].values])
        texts = [row_text(c, feature_name, t, d) for c, t, d in zip(train_sub['category'], train_sub['title'], train_sub['description'])]
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
        X_train = vectorizer.fit_transform(texts)
        svd_dim = max(16, min(128, X_train.shape[0] - 1, X_train.shape[1] - 1))
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        X_train_red = svd.fit_transform(X_train)
        model = CatBoostClassifier(
            iterations=180,
            learning_rate=0.08,
            depth=8,
            loss_function='MultiClass',
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train_red, y)
        eval_sub = eval_df.loc[feature_rows.index]
        X_eval = svd.transform(vectorizer.transform(eval_sub['text']))
        probs = model.predict_proba(X_eval)
        for row_idx, prob_row, allowed_values in zip(eval_sub.index, probs, eval_sub['allowed_values']):
            value, score, margin = argmax_allowed(prob_row, labels, allowed_values)
            pos = eval_df.index.get_loc(row_idx)
            all_candidate_preds[pos] = value
            all_candidate_scores[pos] = score
            all_candidate_margins[pos] = margin
    best = None
    for min_prob in (0.35, 0.45, 0.55, 0.65, 0.75):
        for min_margin in (0.00, 0.05, 0.10, 0.15, 0.20):
            stats = evaluate_predictions(eval_df, all_candidate_preds, all_candidate_scores, all_candidate_margins, min_prob, min_margin)
            stats['min_prob'] = min_prob
            stats['min_margin'] = min_margin
            if best is None or stats['combined_acc'] > best['combined_acc'] or (stats['combined_acc'] == best['combined_acc'] and stats['accepted_acc'] > best['accepted_acc']):
                best = stats
    best['method'] = 'catboost_tfidf_svd'
    return best


def benchmark_transformer(eval_df, train_joined_df):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

    feature_scores = []
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    usable_eval = eval_df[eval_df['feature_name'].isin(TRANSFORMER_FEATURES)].copy()
    if usable_eval.empty:
        return {'method': 'local_sequence_classifier', 'accepted': 0, 'accepted_rate': 0.0, 'accepted_acc': 0.0, 'combined_acc': baseline_accuracy(eval_df), 'note': 'no eval rows'}

    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.enc = tokenizer(texts, truncation=True, padding=True, max_length=192)
            self.labels = labels
        def __len__(self):
            return len(self.enc['input_ids'])
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx])
            return item

    all_candidate_preds = [None] * len(eval_df)
    all_candidate_scores = [None] * len(eval_df)
    all_candidate_margins = [None] * len(eval_df)

    for feature_name, feature_rows in usable_eval.groupby('feature_name', sort=False):
        train_sub = train_joined_df[train_joined_df['feature_name'] == feature_name][['category', 'feature_name', 'title', 'description', 'feature_value']].copy()
        if len(train_sub) < 100 or train_sub['feature_value'].nunique() < 2 or train_sub['feature_value'].nunique() > 20:
            continue
        train_sub = sample_training_rows(train_sub, max_rows=4000, max_per_label=500)
        labels = sorted(train_sub['feature_value'].unique())
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        y = [label_to_id[value] for value in train_sub['feature_value'].values]
        texts = [row_text(c, feature_name, t, d) for c, t, d in zip(train_sub['category'], train_sub['title'], train_sub['description'])]
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, num_labels=len(labels), ignore_mismatched_sizes=True, local_files_only=True)
        train_ds = TextDataset(texts, y)
        args = TrainingArguments(
            output_dir=os.path.join('tmp_seqcls', feature_name.replace('/', '_').replace(' ', '_')),
            per_device_train_batch_size=16,
            num_train_epochs=1,
            learning_rate=3e-5,
            logging_strategy='no',
            save_strategy='no',
            eval_strategy='no',
            report_to='none',
            disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer)
        trainer.train()
        eval_sub = eval_df.loc[feature_rows.index]
        eval_ds = TextDataset(eval_sub['text'].tolist(), None)
        pred_output = trainer.predict(eval_ds)
        logits = pred_output.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        for row_idx, prob_row, allowed_values in zip(eval_sub.index, probs, eval_sub['allowed_values']):
            value, score, margin = argmax_allowed(prob_row, labels, allowed_values)
            pos = eval_df.index.get_loc(row_idx)
            all_candidate_preds[pos] = value
            all_candidate_scores[pos] = score
            all_candidate_margins[pos] = margin

    best = None
    for min_prob in (0.40, 0.50, 0.60, 0.70, 0.80):
        for min_margin in (0.00, 0.05, 0.10, 0.15, 0.20):
            stats = evaluate_predictions(eval_df, all_candidate_preds, all_candidate_scores, all_candidate_margins, min_prob, min_margin)
            stats['min_prob'] = min_prob
            stats['min_margin'] = min_margin
            if best is None or stats['combined_acc'] > best['combined_acc'] or (stats['combined_acc'] == best['combined_acc'] and stats['accepted_acc'] > best['accepted_acc']):
                best = stats
    best['method'] = 'local_sequence_classifier'
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=10000)
    args = parser.parse_args()

    start = time.time()
    report, train_joined_df, _ = build_eval_report(args.sample_size)
    fallback_df = report[(report['stage'] == 'categorical_fallback_global') & (report['feature_name'].isin(TARGET_FEATURES))].copy()
    print('sample_size', args.sample_size)
    print('target residual rows', len(fallback_df))
    print('baseline_acc', round(baseline_accuracy(fallback_df), 4))
    print('feature_counts')
    print(fallback_df['feature_name'].value_counts().to_string())
    print('---')
    for fn in (benchmark_sklearn, benchmark_lgbm, benchmark_catboost, benchmark_transformer):
        method_start = time.time()
        try:
            result = fn(fallback_df, train_joined_df)
            result['seconds'] = round(time.time() - method_start, 2)
            print(result)
        except Exception as exc:
            print({'method': fn.__name__, 'error': repr(exc), 'seconds': round(time.time() - method_start, 2)})
    print('total_seconds', round(time.time() - start, 2))

if __name__ == '__main__':
    main()
