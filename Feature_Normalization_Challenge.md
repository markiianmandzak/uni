# Feature Normalization Challenge

## Product Title + Description → Normalized Taxonomy Features

### 1. The Mission
In real-world product catalogs, many relevant attributes are not cleanly structured. Instead, they are hidden in titles and descriptions—phrased differently, using abbreviations, variations, and sometimes inconsistent data quality. Unite has around 200 million such products.

Your hackathon challenge: Within **2.5 days**, develop an approach that turns **unstructured product text** into a set of **normalized features**—mapped to a **given taxonomy**. The goal is an outcome that is **measurable** and can be **fairly compared** with other teams in a standardized evaluation setup.

---

## 2. Setting (What you get, what you deliver)
You work in a clearly defined setup:

- **Input:** per product: `uid`, `category`, `title`, `description`
- **Target space:** `taxonomy.parquet` defines valid `feature_name` and `feature_type` per `category`
- **Output:** one submission file with predicted feature rows  `submission.parquet` 
- **Evaluation:** via the provided local evaluator (`evaluate.py`)

---

## 3. Task
For each product (`uid`) in a split, predict a set of normalized feature rows:

- Use `title` and `description` as the text signal.
- Use `category` as context to select the relevant features from the taxonomy.
- Output feature names and feature types **exactly as defined in the taxonomy**.
- For **categorical** features, values must be consistent with the allowed values in `aggregated_feature_values`.
- If the attribute type is **numeric**, you **MUST** use the unit and format of the taxonomy; some examples are provided in `aggregated_feature_values`; the numeric value may vary.

The solution approach is intentionally open: rules/regex, classic NLP, ML models, local LLMs, hybrid approaches—everything is allowed, as long as your submission matches the specifications and can be evaluated locally in a reproducible way.

---

## 4. Provided Data and Evaluation Code
This repo contains:

- `evaluate.py` — local evaluator
- `data/` — Parquet files (train/val/test + taxonomy)

---

## 5. Data Layout
The provided folder structure:

| Path | Content |
|---|---|
| `data/taxonomy.parquet` | Taxonomy per category: `category`,`feature_name`, `feature_type` (`categorical\|numeric`), `aggregated_feature_values` (for categorical) |
| `data/train/products.parquet` | Train products: `uid`, `category`, `title`, `description` |
| `data/train/product_features.parquet` | Train ground truth: `uid`, `feature_name`, `feature_value`, `feature_type` |
| `data/val/products.parquet` | Val products: `uid`, `category`, `title`, `description` |
| `data/val/product_features.parquet` | Val ground truth: `uid`, `feature_name`, `feature_value`, `feature_type` |
| `data/test/products.parquet` | Test products: `uid`, `category`, `title`, `description` |
| `data/test/submission.parquet` | Test Predictions (`feature_value` is NULL): `uid`, `feature_name`, `feature_value`, `feature_type` |
---

## 6. Results Submission 
Submit exactly **one** file:

- `submission.parquet` 

It must contain these columns **exactly** (case-sensitive):

- `uid`           <not predicted: comes from `test/products.parquet`>
- `feature_name`  <not predicted: comes from `taxonomy.parquet`, for the product’s `category`>
- `feature_value` <**Predicted Solution: the value has to come from YOUR method, with constraints mentioned under 3. Task.**>
- `feature_type`  <not predicted: comes from `taxonomy.parquet`,  for the product’s `category`>

Note: For complete deliverables, see section 10.

---

## 7. Evaluation (Local)
Example: evaluate a Parquet submission on validation:

```bash
python evaluate.py --split val --data_dir data --pred_path submission.parquet
```

---

The evaluator uses **Exact Value Match Accuracy** for a given (`uid`, `feature_name`) as the metric.

---

## 8. Rules & Constraints
- Strictly Not Allowed: Asking LLM for every product and scraping any information from mercateo website
- Do not modify the provided parquet files.
- Predictions must follow the submission schema exactly.
- Only taxonomy-defined feature names are valid for a product’s category.

---

## 9. Common failure cases (avoid these)
- Missing required columns in the submission
- Wrong column names (case-sensitive)
- Predicting features that don’t exist for the product’s category
- Categorical values that don’t match the taxonomy’s allowed values format
- Numeric feature values inconsistent with the label style (units/format)

---

## 10. Deliverables (Required)
1. `submission.parquet` in the required schema
2. Short doc (max. 1–2 pages): approach (high-level), assumptions, limitations, next steps
3. Short run guide: how we can reproduce your approach locally
4. Short demo (5 minutes): idea → pipeline → output → score on `val`

---

## 11. Evaluation Rubric (3 criteria)
Winners are determined using these **three criteria**:

1. **[40%] Exact Value Match Accuracy**
2. **[20%] Throughput and scalability**
3. **[40%] Total cost = one-time cost + recurring cost**
   - If using LLMs, costs must include token usage required to generate results at scale (e.g., for **X million products**).

**Reproducibility requirement:** Results must be easy to reproduce. In particular, if using LLMs, all statements about token usage must be verifiable (e.g., via logs/usage reports and a reproducible run configuration).

---

## 12. Note on Ambiguities
If something is unclear ask the organizers.