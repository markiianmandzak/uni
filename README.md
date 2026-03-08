# Feature Normalization Challenge - Product Taxonomy Extraction

A comprehensive solution for extracting and normalizing product features from unstructured text (titles and descriptions) into a structured taxonomy. This project was developed for a hackathon challenge to process ~200 million products with measurable, reproducible results.

## 🎯 Challenge Overview

**Mission**: Transform unstructured product text into normalized features mapped to a given taxonomy.

**Input**: Product data with `uid`, `category`, `title`, `description`  
**Output**: Normalized feature rows with `feature_name`, `feature_value`, `feature_type`  
**Evaluation**: Exact Value Match Accuracy on validation/test sets

### Evaluation Criteria
1. **[40%] Exact Value Match Accuracy** - Precision of extracted features
2. **[20%] Throughput and Scalability** - Processing speed at scale
3. **[40%] Total Cost** - One-time + recurring costs (including LLM token usage)

## 🏗️ Architecture

This solution employs a **hybrid multi-stage pipeline** combining:
- **Rule-based extraction** (regex patterns)
- **Frequency-based matching** (statistical patterns from training data)
- **Machine learning** (fine-tuned T5 model)
- **Ensemble methods** (combining multiple approaches)

### Solution Evolution

The project evolved through 14+ iterations, with each version improving accuracy:

| Version | Accuracy | Key Innovation |
|---------|----------|----------------|
| regex_1 | 47.12% | Basic pattern matching |
| regex_4 | 72.12% | Dimension extraction |
| regex_9 | 94.04% | Special categorical rules |
| regex_14 | **98.69%** | Comprehensive hybrid approach |

## 📁 Project Structure

```
.
├── regex_14.py                    # Main production extractor (98.69% accuracy)
├── src/                           # Modular implementation
│   ├── build_lookup_table.py     # Frequency-based lookup construction
│   ├── frequency_matching.py     # Statistical pattern matching
│   ├── finetune_t5.py           # T5 model fine-tuning
│   ├── hybrid_pipeline.py        # Combined approach orchestration
│   ├── learn_regex_patterns.py   # Pattern mining from training data
│   ├── ensemble_extractor.py     # Multi-model ensemble
│   └── evaluate*.py              # Various evaluation scripts
├── data/                          # Dataset (not in repo)
│   ├── taxonomy/                 # Feature taxonomy definitions
│   ├── train/                    # Training products + features
│   ├── val/                      # Validation set
│   └── test/                     # Test set (predictions only)
├── flan-t5-features/             # Fine-tuned T5 model checkpoints
├── Feature_Normalization_Challenge.md  # Original challenge description
├── KAGGLE_SETUP.md               # GPU inference guide
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas pyarrow tqdm transformers torch
```

### Running the Main Extractor

```bash
# Validate on validation set
python regex_14.py --split val

# Generate test predictions
python regex_14.py --split test

# Sample validation (faster testing)
python regex_14.py --split val --sample-size 10000 --seed 42
```

### Output

- `submission_val.parquet` or `submission_test.parquet` with predictions
- Console output showing:
  - Overall coverage and accuracy
  - Stage-by-stage funnel analysis
  - Cumulative performance metrics

## 🔍 How regex_14.py Works

### Core Architecture

The extractor uses a **multi-stage extraction pipeline** with fallback mechanisms:

```
Input Text → Normalization → Feature Extraction → Value Matching → Output
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              Categorical                          Numeric
                    │                                   │
        ┌───────────┴───────────┐         ┌────────────┴────────────┐
        │                       │         │                         │
   Special Rules          Pattern Match   Dimension            Direct Match
   Mined Aliases          Fuzzy Match     Packaging            Measurement
   Title Priority         Fallback        Thread/Range         Fallback
```

### Key Components

#### 1. Text Normalization (`normalize_text`)
- Converts to lowercase
- Standardizes special characters (×→x, ∅→ø)
- Removes HTML tags
- Normalizes whitespace

#### 2. Taxonomy Rules (`build_taxonomy_rules`)
Builds extraction rules from taxonomy:
- **Categorical features**: Compiles regex patterns for each allowed value
- **Numeric features**: Creates lookup tables with unit normalization
- **Fuzzy matching**: Enables flexible matching for materials, colors, etc.

#### 3. Feature Extraction Strategies

**Categorical Features:**
- **Special Rules**: Hardcoded patterns for complex features (Material, Color, etc.)
- **Mined Aliases**: Statistical patterns learned from training data
- **Pattern Matching**: Direct regex matching with fuzzy options
- **Title Priority**: Prefers matches in product titles
- **Fallback**: Uses most frequent value from training data

**Numeric Features:**
- **Dimension Extraction**: Parses measurements like "100 x 50 x 25 mm"
- **Packaging Units**: Extracts quantities (VE, Stück, Rolle)
- **Thread Specifications**: Handles M8, ST 4.8, G 1/4" formats
- **Range Values**: Extracts min/max from ranges
- **Unit Normalization**: Converts between mm/cm/m, g/kg, etc.

#### 4. Advanced Pattern Recognition

**Tuple Extraction** (`extract_tuple_candidates`):
```python
# Recognizes patterns like:
"H x B x T 100x50x25mm"  → {H: 100mm, B: 50mm, T: 25mm}
"Ø x L 8x20"             → {D: 8mm, L: 20mm}
"L x B 210 x 297 mm"     → {L: 210mm, B: 297mm}
```

**Measurement Extraction** (`extract_measurements`):
```python
# Extracts all numeric values with units:
"Länge 100mm, Breite 50cm" → [(pos, 100, 'mm'), (pos, 50, 'cm')]
```

**Unit Canonicalization**:
- Normalizes unit variations: `g/cm³` → `g/ml`, `v` → `V`
- Converts to base units for comparison
- Handles unit families (length, weight, volume)

#### 5. Mined Alias System (`build_mined_categorical_aliases`)

Learns categorical value patterns from training data:
```python
# Example: "edelstahl a4" appears 98% with value "Edelstahl (A4)"
# Creates pattern: "edelstahl a4" → "Edelstahl (A4)"
```

Requirements for mined patterns:
- Minimum 3 occurrences
- Minimum 98% precision
- Phrase length ≥ 4 characters
- Not purely numeric

#### 6. Fallback Mechanisms

Multi-level fallback strategy:
1. **Category + Feature Mode**: Most common value for this category/feature
2. **Feature Mode**: Most common value for this feature globally
3. **First Allowed Value**: Default to first taxonomy value

### Performance Optimization

**Caching Strategy**:
- Product text normalization cached per UID
- Measurements and tuples extracted once per product
- Regex patterns compiled once and reused

**Processing Order**:
1. Categorical features first (faster, simpler)
2. Numeric features second (more complex)
3. Batch processing with progress tracking

### Stage Reporting

The extractor tracks extraction stages for analysis:

```
categorical_special       - Special hardcoded rules
categorical_alias         - Mined alias patterns
categorical_title_unique  - Single match in title
categorical_text_unique   - Single match in description
categorical_title_multi   - Multiple matches, title preferred
categorical_text_multi    - Multiple matches in text
categorical_fallback      - Default value used

numeric_dimension         - Dimension extraction (L/B/H/D)
numeric_packaging         - Packaging unit extraction
numeric_thread            - Thread specification
numeric_range             - Range value extraction
numeric_format            - Format specification
numeric_direct            - Direct pattern match
numeric_measurement       - General measurement
numeric_fallback          - Default value used
```

## 🧪 Alternative Approaches

### Frequency Matching (`src/frequency_matching.py`)
Statistical approach using training data patterns:
- Builds lookup tables of text→value mappings
- Uses TF-IDF for similarity matching
- Fast but limited to seen patterns

### Fine-tuned T5 (`src/finetune_t5.py`)
Machine learning approach:
- Fine-tunes FLAN-T5 on training data
- Handles unseen patterns better
- Slower and more expensive (GPU required)
- See `KAGGLE_SETUP.md` for GPU inference

### Ensemble Methods (`src/ensemble_extractor.py`)
Combines multiple approaches:
- Voting mechanism across extractors
- Confidence-based selection
- Best for maximizing accuracy

### Hybrid Pipeline (`src/hybrid_pipeline.py`)
Orchestrates multiple methods:
1. Try regex extraction first (fast)
2. Fall back to frequency matching
3. Use T5 for difficult cases
4. Ensemble for final decision

## 📊 Results

### regex_14.py Performance

**Validation Set**:
- Overall Accuracy: **98.69%**
- Coverage: **100%**
- Processing Speed: ~1000 products/second (CPU)

**Stage Breakdown** (example):
```
categorical_special       count=  45231  acc=0.9912  cumulative=  45231  acc=0.9912
categorical_alias         count=  12456  acc=0.9845  cumulative=  57687  acc=0.9891
categorical_title_unique  count=  89234  acc=0.9923  cumulative= 146921  acc=0.9901
numeric_dimension         count=  34567  acc=0.9867  cumulative= 181488  acc=0.9895
numeric_packaging         count=   8901  acc=0.9756  cumulative= 190389  acc=0.9889
...
categorical_fallback      count=  15234  acc=0.8234  cumulative= 250000  acc=0.9869
```

### Cost Analysis

**One-time Costs**:
- Development time: 2.5 days (hackathon)
- Training data analysis: Included in training set
- Pattern mining: ~5 minutes on CPU

**Recurring Costs**:
- CPU-only inference: $0 (uses standard compute)
- No API calls or LLM tokens required
- Scales linearly with product count

**Scalability**:
- 200M products @ 1000/sec = ~55 hours on single CPU
- Easily parallelizable across multiple workers
- No external dependencies or rate limits

## 🔧 Advanced Features

### Custom Pattern Rules

Add domain-specific patterns in `SPECIAL_RULES`:

```python
SPECIAL_RULES = {
    "Material": [
        (re.compile(r'(?i)stahl\s*verzinkt'), 'Stahl verzinkt'),
        (re.compile(r'(?i)edelstahl\s*a4'), 'Edelstahl (A4)'),
        # Add more patterns...
    ],
}
```

### Feature Aliases

Define alternative names for features:

```python
FEATURE_ALIASES = {
    "Länge": ["länge", "gesamtlänge", "l", "lang"],
    "Durchmesser": ["durchmesser", "ø", "dm"],
    # Add more aliases...
}
```

### Unit Conversion

Extend unit families for automatic conversion:

```python
UNIT_FACTORS = {
    "length": {"µm": 0.001, "mm": 1.0, "cm": 10.0, "m": 1000.0},
    "weight": {"mg": 0.001, "g": 1.0, "kg": 1000.0},
    # Add more unit families...
}
```

## 🐛 Troubleshooting

### Low Accuracy for Specific Feature

1. Check if feature has taxonomy rules:
   ```python
   rule = taxonomy_rules.get((category, feature_name))
   ```

2. Add special extraction logic:
   ```python
   def extract_my_feature_value(text, rule):
       # Custom extraction logic
       pass
   ```

3. Mine aliases from training data:
   ```python
   # Ensure feature is in ALIAS_FEATURES
   ALIAS_FEATURES.add("MyFeature")
   ```

### Missing Values

- Check if values are in taxonomy: `rule["values"]`
- Verify text normalization isn't removing key information
- Add fuzzy matching: `FUZZY_CATEGORICAL_FEATURES.add("MyFeature")`

### Performance Issues

- Use `--sample-size` for faster testing
- Profile with `python -m cProfile regex_14.py`
- Consider parallel processing for large datasets

## 📚 Key Learnings

1. **Hybrid approaches win**: Combining rules, statistics, and ML beats any single method
2. **Domain knowledge matters**: Hardcoded rules for complex patterns (threads, materials) are essential
3. **Training data is gold**: Mining patterns from training data provides high-precision extraction
4. **Fallbacks are critical**: Always have a default value to maintain 100% coverage
5. **Unit normalization is hard**: Handling mm/cm/m, different formats, and edge cases requires careful design

## 🤝 Contributing

This was a hackathon project, but improvements are welcome:
- Add more special rules for specific features
- Improve numeric extraction patterns
- Optimize performance for large-scale processing
- Add support for new feature types

## 📄 License

Hackathon project - check with organizers for usage rights.

## 🙏 Acknowledgments

- Challenge organizers for the well-structured problem
- Training data providers
- Open-source libraries: pandas, transformers, torch

---

**For questions or issues, please refer to the original challenge documentation in `Feature_Normalization_Challenge.md`**
