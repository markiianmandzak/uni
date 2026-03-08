"""
Categorical Feature Extractor — High-quality parser for categorical features.
Handles bracket stripping, boundary-aware matching, fuzzy matching,
training priors, and multi-strategy extraction.

Target: >60% exact match accuracy on validation set (categoricals only).
"""

import pandas as pd
import numpy as np
import re
import time
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

PYTHON = True  # marker

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 80)
print("CATEGORICAL FEATURE EXTRACTOR")
print("=" * 80)

t_total = time.time()

products_train = pd.read_parquet("data/train/products.parquet")
features_train = pd.read_parquet("data/train/product_features.parquet")
products_val = pd.read_parquet("data/val/products.parquet")
features_val = pd.read_parquet("data/val/product_features.parquet")
taxonomy = pd.read_parquet("data/taxonomy/taxonomy.parquet")

print(f"Train: {len(products_train)} products, {len(features_train)} features")
print(f"Val:   {len(products_val)} products, {len(features_val)} features")
print(f"Taxonomy: {len(taxonomy)} rows")

# =============================================================================
# STEP 1: CREATE SHUFFLED TRAINING DATA (if not exists)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: CREATING SHUFFLED TRAINING DATA")
print("=" * 80)

import os
shuf_prod_path = "data/train/products.shuffled.parquet"
shuf_feat_path = "data/train/product_features.shuffled.parquet"

if os.path.exists(shuf_prod_path) and os.path.exists(shuf_feat_path):
    print("Shuffled files already exist, loading...")
    products_train = pd.read_parquet(shuf_prod_path)
    features_train = pd.read_parquet(shuf_feat_path)
else:
    print("Creating shuffled versions...")
    products_train = products_train.sample(frac=1, random_state=42).reset_index(drop=True)
    features_train = features_train.sample(frac=1, random_state=42).reset_index(drop=True)
    products_train.to_parquet(shuf_prod_path)
    features_train.to_parquet(shuf_feat_path)
    print(f"Saved shuffled data.")

print(f"Shuffled Train: {len(products_train)} products, {len(features_train)} features")

# =============================================================================
# STEP 2: FILTER TO CATEGORICALS ONLY
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: ANALYZING CATEGORICAL FEATURES")
print("=" * 80)

cat_features_train = features_train[features_train['feature_type'] == 'categorical'].copy()
cat_features_val = features_val[features_val['feature_type'] == 'categorical'].copy()
cat_taxonomy = taxonomy[taxonomy['feature_type'] == 'categorical'].copy()

print(f"Categorical - Train: {len(cat_features_train)}, Val: {len(cat_features_val)}, Tax: {len(cat_taxonomy)}")

# =============================================================================
# PARSE TAXONOMY ALLOWED VALUES
# =============================================================================
print("\nParsing taxonomy allowed values...")


def parse_aggregated_values(agg_str):
    """Parse taxonomy aggregated_feature_values string.
    Format: {'[Stahl]','[Edelstahl (A2)]','[A4]'}
    Returns list of clean values with brackets stripped.
    """
    if not agg_str or pd.isna(agg_str):
        return []

    agg_str = str(agg_str).strip()
    if agg_str.startswith('{') and agg_str.endswith('}'):
        agg_str = agg_str[1:-1]

    values = []
    # Match [value] possibly wrapped in quotes
    for m in re.finditer(r"'?\[([^\]]*)\]'?", agg_str):
        val = m.group(1).strip()
        if val:
            values.append(val)

    if not values:
        # Fallback: split by comma
        parts = agg_str.split("','")
        for p in parts:
            p = p.strip().strip("'").strip('[]').strip()
            if p:
                values.append(p)

    return values


# Build taxonomy lookup: (category, feature_name) -> list of allowed values
tax_values = {}
for _, row in cat_taxonomy.iterrows():
    key = (row['category'], row['feature_name'])
    vals = parse_aggregated_values(row['aggregated_feature_values'])
    tax_values[key] = vals

print(f"Taxonomy categorical pairs: {len(tax_values)}")

# =============================================================================
# BUILD TRAINING PRIORS
# =============================================================================
print("Building training priors from shuffled data...")

# Merge train features with product categories
train_merged = cat_features_train.merge(
    products_train[['uid', 'category']], on='uid', how='left'
)

# Category-specific prior: (category, feature_name) -> {value: count}
feat_priors_cat = {}
for (cat, fname), grp in train_merged.groupby(['category', 'feature_name']):
    feat_priors_cat[(cat, fname)] = grp['feature_value'].value_counts().to_dict()

# Global prior: feature_name -> {value: count}
feat_priors_global = {}
for fname, grp in train_merged.groupby('feature_name'):
    feat_priors_global[fname] = grp['feature_value'].value_counts().to_dict()

print(f"Category-feature priors: {len(feat_priors_cat)}")
print(f"Global feature priors: {len(feat_priors_global)}")

# =============================================================================
# TEXT CLEANING
# =============================================================================
_tag_re = re.compile(r'<[^>]+>')
_space_re = re.compile(r'\s+')
_special_chars = re.compile(r'[®™©]')
_token_re = re.compile(r'[a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß0-9\-\+\.]*[a-zA-ZäöüÄÖÜß0-9]|[a-zA-ZäöüÄÖÜß0-9]')


def clean_text(text):
    """Clean HTML tags and normalize whitespace."""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = _tag_re.sub(' ', text)
    text = _special_chars.sub(' ', text)
    text = _space_re.sub(' ', text)
    return text.strip()


# =============================================================================
# CATEGORICAL VALUE MATCHER
# =============================================================================
class CategoricalMatcher:
    """Multi-strategy matcher for categorical feature values.

    Strategies (priority order):
    1. Single allowed value -> return immediately
    2. Exact boundary match in title (word boundaries)
    3. Exact boundary match in description
    4. Loose substring match in title (no boundaries)
    5. Loose substring match in description
    6. Tight match (strip spaces/hyphens) in title/desc
    7. Umlaut normalization match
    8. Partial token match for multi-word values
    9. Category-name derivation
    10. Training prior (category-specific -> global)
    11. Fallback: first allowed value
    """

    def __init__(self, allowed_values, feature_name, cat_priors, global_priors, keyword_scores=None):
        self.allowed = allowed_values
        self.feature_name = feature_name
        self.cat_priors = cat_priors or {}
        self.global_priors = global_priors or {}
        self.keyword_scores = keyword_scores or {}

        # Sort allowed values longest first for greedy matching
        self.allowed_sorted = sorted(allowed_values, key=len, reverse=True)

        # Pre-compile boundary-aware regex for each value (longest first)
        self.boundary_patterns = []
        for v in self.allowed_sorted:
            vl = v.lower()
            try:
                # Word boundary that handles German umlauts and numbers
                pat = re.compile(
                    r'(?<![a-zA-Z0-9äöüÄÖÜß])' +
                    re.escape(vl) +
                    r'(?![a-zA-Z0-9äöüÄÖÜß])',
                    re.IGNORECASE
                )
                self.boundary_patterns.append((v, vl, pat))
            except re.error:
                self.boundary_patterns.append((v, vl, None))

        # Pre-compute tight forms (no spaces, hyphens, underscores)
        self.tight_forms = []
        for v in self.allowed_sorted:
            tight = re.sub(r'[\s\-_]', '', v.lower())
            if len(tight) >= 2:
                self.tight_forms.append((v, tight))

        # Pre-compute umlaut-normalized forms
        self.umlaut_forms = []
        for v in self.allowed_sorted:
            vl = v.lower()
            # ä->ae, ö->oe, ü->ue, ß->ss
            norm = vl.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
            if norm != vl:
                self.umlaut_forms.append((v, norm))
            # reverse: ae->ä etc.
            rev = vl.replace('ae', 'ä').replace('oe', 'ö').replace('ue', 'ü').replace('ss', 'ß')
            if rev != vl:
                self.umlaut_forms.append((v, rev))

        # Pre-compute token sets for multi-word values
        self.token_sets = []
        for v in self.allowed_sorted:
            tokens = set(re.findall(r'[a-zA-ZäöüÄÖÜß0-9]+', v.lower()))
            if len(tokens) >= 2 and all(len(t) >= 2 for t in tokens):
                self.token_sets.append((v, tokens))

        # Pre-compute parenthetical patterns: "main (qualifier)"
        self.paren_forms = []
        for v in allowed_values:
            m = re.match(r'^(.+?)\s*\((.+?)\)$', v)
            if m:
                main = m.group(1).strip().lower()
                qual = m.group(2).strip().lower()
                if len(main) >= 2 and len(qual) >= 1:
                    self.paren_forms.append((v, main, qual))

        # Pre-compute slash patterns: "val1 / val2"
        self.slash_forms = []
        for v in allowed_values:
            if '/' in v:
                parts = [p.strip().lower() for p in v.split('/') if len(p.strip()) >= 2]
                if len(parts) >= 2:
                    self.slash_forms.append((v, parts))

        # Pre-compute paren extensions: map base -> [(full_value, qualifier)]
        # e.g. "Edelstahl" -> [("Edelstahl (A2)", "a2"), ("Edelstahl (A4)", "a4")]
        self.paren_extensions = defaultdict(list)
        for v, main, qual in self.paren_forms:
            self.paren_extensions[main].append((v, qual))

        # Pre-compute containment hierarchy: value -> [longer values containing it]
        # e.g. "Stahl" -> ["Edelstahl", "Chrom-Vanadium-Stahl"]
        self.longer_values = defaultdict(list)
        for v in allowed_values:
            vl = v.lower()
            for v2 in allowed_values:
                v2l = v2.lower()
                if v2l != vl and vl in v2l and len(v2l) > len(vl):
                    self.longer_values[vl].append(v2)

    def _pick_best(self, candidates, title_lower=None):
        """Pick best from multiple matches.
        
        Strategy:
        1. If any candidate fully contains another, prefer the contained-in (longer/more specific)
        2. Among equally nested candidates, use training priors
        3. Else pick first-appearing in title if available
        """
        if len(candidates) == 1:
            return candidates[0]

        # Check containment: remove values that are substrings of other candidates
        # e.g. if both "Stahl" and "Edelstahl" matched, drop "Stahl"
        candidates_lower = [(c, c.lower()) for c in candidates]
        non_contained = []
        for c, cl in candidates_lower:
            is_contained = False
            for c2, c2l in candidates_lower:
                if c2l != cl and cl in c2l and len(c2l) > len(cl):
                    is_contained = True
                    break
            if not is_contained:
                non_contained.append(c)

        if len(non_contained) == 1:
            return non_contained[0]
        if not non_contained:
            non_contained = candidates  # fallback

        # Category-specific prior
        if self.cat_priors:
            best = max(non_contained, key=lambda c: self.cat_priors.get(c, 0))
            if self.cat_priors.get(best, 0) > 0:
                return best

        # Global prior
        if self.global_priors:
            best = max(non_contained, key=lambda c: self.global_priors.get(c, 0))
            if self.global_priors.get(best, 0) > 0:
                return best

        # Position-based: first occurring in title
        if title_lower:
            best_pos = len(title_lower) + 1
            best_val = non_contained[0]
            for c in non_contained:
                pos = title_lower.find(c.lower())
                if 0 <= pos < best_pos:
                    best_pos = pos
                    best_val = c
            if best_pos < len(title_lower) + 1:
                return best_val

        # Fallback to longest (most specific)
        return max(non_contained, key=len)

    def _refine_with_paren(self, base_val, title_lower, full_text_lower):
        """Upgrade a base match to a more specific form if evidence is in text.
        
        E.g., base='Edelstahl' -> 'Edelstahl (A2)' if 'a2' found in text.
        Also: base='Stahl' -> 'Chrom-Vanadium-Stahl' if longer value found.
        """
        base_lower = base_val.lower()

        # 1. Paren extension: "Edelstahl" -> "Edelstahl (A2)"
        extensions = self.paren_extensions.get(base_lower, [])
        if extensions:
            title_matches = []
            full_matches = []
            for full_val, qual in extensions:
                try:
                    qual_pat = re.compile(
                        r'(?<![a-zA-Z0-9äöüÄÖÜß])' + re.escape(qual) + r'(?![a-zA-Z0-9äöüÄÖÜß])',
                        re.IGNORECASE
                    )
                    if qual_pat.search(title_lower):
                        title_matches.append(full_val)
                    elif qual_pat.search(full_text_lower):
                        full_matches.append(full_val)
                except re.error:
                    if qual in title_lower:
                        title_matches.append(full_val)
                    elif qual in full_text_lower:
                        full_matches.append(full_val)

            if title_matches:
                return self._pick_best(title_matches, title_lower)
            if full_matches:
                return self._pick_best(full_matches, title_lower)

        # 2. Containment upgrade: "Stahl" -> "Chrom-Vanadium-Stahl"
        longer_vals = self.longer_values.get(base_lower, [])
        if longer_vals:
            upgrade_title = []
            upgrade_full = []
            for lv in longer_vals:
                lvl = lv.lower()
                try:
                    pat = re.compile(
                        r'(?<![a-zA-Z0-9äöüÄÖÜß])' + re.escape(lvl) + r'(?![a-zA-Z0-9äöüÄÖÜß])',
                        re.IGNORECASE
                    )
                    if pat.search(title_lower):
                        upgrade_title.append(lv)
                    elif pat.search(full_text_lower):
                        upgrade_full.append(lv)
                except re.error:
                    pass

            if upgrade_title:
                return self._pick_best(upgrade_title, title_lower)
            if upgrade_full:
                return self._pick_best(upgrade_full, title_lower)

            # Also check tight match for longer values
            tight_title = re.sub(r'[\s\-_]', '', title_lower)
            tight_full = re.sub(r'[\s\-_]', '', full_text_lower)
            for lv in longer_vals:
                tight_lv = re.sub(r'[\s\-_]', '', lv.lower())
                if len(tight_lv) >= 5:
                    if tight_lv in tight_title:
                        return lv
                    if tight_lv in tight_full:
                        return lv

        return base_val

    def _best_prior(self, allowed_set=None):
        """Return the most common value from training priors."""
        # Category-specific
        if self.cat_priors:
            for val, _ in sorted(self.cat_priors.items(), key=lambda x: -x[1]):
                if allowed_set is None or val in allowed_set:
                    return val, 'prior_cat'

        # Global
        if self.global_priors:
            for val, _ in sorted(self.global_priors.items(), key=lambda x: -x[1]):
                if allowed_set is None or val in allowed_set:
                    return val, 'prior_global'

        return None, None

    def match(self, full_text_lower, title_lower, category_lower):
        """Try all strategies. Returns (value, method) or (None, None)."""

        # --- Strategy 1: Single allowed value ---
        if len(self.allowed) == 1:
            return self.allowed[0], 'single_value'

        # --- Strategy 2-3: Boundary match in title, then desc ---
        title_hits = []
        desc_hits = []
        for v, vl, pat in self.boundary_patterns:
            if pat is not None:
                if pat.search(title_lower):
                    title_hits.append(v)
                elif pat.search(full_text_lower):
                    desc_hits.append(v)

        if title_hits:
            best = self._pick_best(list(dict.fromkeys(title_hits)), title_lower)
            # Paren refinement: if best is "Edelstahl" and "Edelstahl (A2)" is available,
            # check if qualifier "a2" also in text
            refined = self._refine_with_paren(best, title_lower, full_text_lower)
            return refined, 'boundary_title'
        if desc_hits:
            best = self._pick_best(list(dict.fromkeys(desc_hits)), title_lower)
            refined = self._refine_with_paren(best, title_lower, full_text_lower)
            return refined, 'boundary_desc'

        # --- Strategy 4: Parenthetical match (BEFORE substr to avoid "Schlitz" matching "Kreuzschlitz") ---
        paren_title = []
        paren_desc = []
        for v, main, qual in self.paren_forms:
            if main in title_lower and qual in title_lower:
                paren_title.append(v)
            elif main in full_text_lower and qual in full_text_lower:
                paren_desc.append(v)

        if paren_title:
            return self._pick_best(list(dict.fromkeys(paren_title))), 'paren_title'
        if paren_desc:
            return self._pick_best(list(dict.fromkeys(paren_desc))), 'paren_desc'

        # --- Strategy 5-6: Loose substring match ---
        title_sub = []
        desc_sub = []
        for v in self.allowed_sorted:
            vl = v.lower()
            if len(vl) >= 3:  # avoid very short false positives
                if vl in title_lower:
                    title_sub.append(v)
                elif vl in full_text_lower:
                    desc_sub.append(v)

        if title_sub:
            best = self._pick_best(list(dict.fromkeys(title_sub)), title_lower)
            refined = self._refine_with_paren(best, title_lower, full_text_lower)
            return refined, 'substr_title'
        if desc_sub:
            best = self._pick_best(list(dict.fromkeys(desc_sub)), title_lower)
            refined = self._refine_with_paren(best, title_lower, full_text_lower)
            return refined, 'substr_desc'

        # --- Strategy 7: Tight match ---
        tight_title = re.sub(r'[\s\-_]', '', title_lower)
        tight_full = re.sub(r'[\s\-_]', '', full_text_lower)
        tight_hits_t = []
        tight_hits_d = []
        for v, tight in self.tight_forms:
            if len(tight) >= 3:
                if tight in tight_title:
                    tight_hits_t.append(v)
                elif tight in tight_full:
                    tight_hits_d.append(v)

        if tight_hits_t:
            return self._pick_best(list(dict.fromkeys(tight_hits_t))), 'tight_title'
        if tight_hits_d:
            return self._pick_best(list(dict.fromkeys(tight_hits_d))), 'tight_desc'

        # --- Strategy 8: Umlaut normalization ---
        for v, norm in self.umlaut_forms:
            if norm in title_lower:
                return v, 'umlaut_title'
            if norm in full_text_lower:
                return v, 'umlaut_desc'

        # --- Strategy 9: Token overlap for multi-word values ---
        title_tokens = set(_token_re.findall(title_lower))
        full_tokens = set(_token_re.findall(full_text_lower))
        token_title = []
        token_desc = []
        for v, tokens in self.token_sets:
            if tokens.issubset(title_tokens):
                token_title.append(v)
            elif tokens.issubset(full_tokens):
                token_desc.append(v)

        if token_title:
            return self._pick_best(list(dict.fromkeys(token_title))), 'token_title'
        if token_desc:
            return self._pick_best(list(dict.fromkeys(token_desc))), 'token_desc'

        # --- Strategy 10: Slash-separated match ---
        for v, parts in self.slash_forms:
            if all(p in title_lower for p in parts):
                return v, 'slash_title'
            if all(p in full_text_lower for p in parts):
                return v, 'slash_desc'

        # --- Strategy 11: Keyword voting from training ---
        if self.keyword_scores:
            best_val = None
            best_score = 0.0
            for v in self.allowed:
                score = 0.0
                v_keywords = self.keyword_scores.get(v)
                if v_keywords:
                    for tok, s in v_keywords.items():
                        if tok in title_tokens:
                            score += s * 2.0  # title weight
                        elif tok in full_tokens:
                            score += s
                if score > best_score:
                    best_score = score
                    best_val = v
            if best_val and best_score > 0.2:
                return best_val, 'keyword_vote'

        # --- Strategy 12: Training prior ---
        allowed_set = set(self.allowed)
        prior_val, prior_method = self._best_prior(allowed_set)
        if prior_val:
            return prior_val, prior_method

        # --- Strategy 13: First allowed value (absolute fallback) ---
        if self.allowed:
            return self.allowed[0], 'fallback_first'

        return None, None


# =============================================================================
# BUILD KEYWORD SCORES FROM TRAINING DATA
# =============================================================================
print("Building keyword scores from training titles...")
t_kw = time.time()

# Merge training features with product titles
train_with_text = train_merged.merge(
    products_train[['uid', 'title']], on='uid', how='left'
)

# For each feature_name, compute keyword distinctiveness per value
# keyword_scores_by_feature[fname][value] = {token: distinctiveness_score}
keyword_scores_by_feature = {}

for fname, grp in train_with_text.groupby('feature_name'):
    # Count tokens per value and globally for this feature
    value_token_counts = defaultdict(Counter)
    total_token_counts = Counter()
    value_total_docs = Counter()
    
    for _, row in grp.iterrows():
        val = row['feature_value']
        title = str(row['title']).lower() if pd.notna(row['title']) else ''
        tokens = set(_token_re.findall(title))
        value_total_docs[val] += 1
        for t in tokens:
            if len(t) >= 2:
                value_token_counts[val][t] += 1
                total_token_counts[t] += 1
    
    # Compute distinctive tokens per value
    value_keywords = {}
    total_docs = sum(value_total_docs.values())
    
    for val, tcounts in value_token_counts.items():
        keywords = {}
        n_docs_val = value_total_docs[val]
        if n_docs_val < 3:
            continue
        
        for t, count in tcounts.items():
            if count < 3:
                continue
            # Specificity: fraction of this token's occurrences that belong to this value
            specificity = count / total_token_counts[t]
            # Frequency: fraction of this value's documents that contain this token
            frequency = count / n_docs_val
            
            # Token is distinctive if it appears mostly with this value
            # and appears in a reasonable fraction of this value's docs
            if specificity > 0.3 and frequency > 0.05:
                # Score combines specificity and frequency
                keywords[t] = specificity * frequency
        
        if keywords:
            # Keep top 30 keywords per value to avoid noise
            top_kw = dict(sorted(keywords.items(), key=lambda x: -x[1])[:30])
            value_keywords[val] = top_kw
    
    if value_keywords:
        keyword_scores_by_feature[fname] = value_keywords

print(f"Keyword scores for {len(keyword_scores_by_feature)} feature names in {time.time() - t_kw:.1f}s")

# =============================================================================
# PRE-COMPILE ALL MATCHERS
# =============================================================================
print("\nPre-compiling categorical matchers...")
t_compile = time.time()

matchers = {}
for (cat, fname), vals in tax_values.items():
    if vals:
        cat_prior = feat_priors_cat.get((cat, fname), {})
        global_prior = feat_priors_global.get(fname, {})
        kw_scores = keyword_scores_by_feature.get(fname, {})
        matchers[(cat, fname)] = CategoricalMatcher(vals, fname, cat_prior, global_prior, kw_scores)

print(f"Compiled {len(matchers)} matchers in {time.time() - t_compile:.1f}s")


# =============================================================================
# PREDICTION FUNCTION (importable)
# =============================================================================
def predict_categorical_row(title_lower, full_lower, feature_name, category):
    """Predict a single categorical feature value. Returns string or None."""
    category_lower = category.lower() if category else ''
    key = (category, feature_name)

    # Try pre-compiled matcher
    matcher = matchers.get(key)
    if matcher:
        predicted, method = matcher.match(full_lower, title_lower, category_lower)
        if predicted is not None:
            return str(predicted).strip()

    # Dynamic fallback for missing matchers
    allowed = tax_values.get(key, [])
    if allowed:
        cat_prior = feat_priors_cat.get(key, {})
        global_prior = feat_priors_global.get(feature_name, {})
        kw_scores = keyword_scores_by_feature.get(feature_name, {})
        temp = CategoricalMatcher(allowed, feature_name, cat_prior, global_prior, kw_scores)
        predicted, method = temp.match(full_lower, title_lower, category_lower)
        if predicted is not None:
            return str(predicted).strip()

    # Global prior (no taxonomy)
    if feature_name in feat_priors_global:
        best = max(feat_priors_global[feature_name].items(), key=lambda x: x[1])[0]
        return str(best).strip()

    return None


if __name__ == '__main__':
    # =============================================================================
    # STEP 3: EVALUATE ON VALIDATION SET
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATING ON VALIDATION SET (CATEGORICALS ONLY)")
    print("=" * 80)

    val_merged = cat_features_val.merge(
        products_val[['uid', 'category', 'title', 'description']],
        on='uid', how='left'
    )
    print(f"Categorical validation rows: {len(val_merged)}")

    # Pre-clean all val texts
    print("Pre-cleaning validation texts...")
    t_clean = time.time()
    uid_text_cache = {}
    for _, row in products_val.iterrows():
        uid = row['uid']
        title_clean = clean_text(row['title'])
        desc_clean = clean_text(row['description'])
        uid_text_cache[uid] = (title_clean.lower(), (title_clean + " " + desc_clean).lower())
    print(f"Cached text for {len(uid_text_cache)} products in {time.time() - t_clean:.1f}s")

    # Run evaluation
    correct = 0
    total = 0
    method_stats = Counter()
    method_correct = Counter()
    feature_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    errors = []

    t0 = time.time()

    for i, (_, row) in enumerate(val_merged.iterrows()):
        uid = row['uid']
        category = row['category']
        feature_name = row['feature_name']
        actual = str(row['feature_value']).strip()

        text_data = uid_text_cache.get(uid)
        if text_data:
            title_lower, full_lower = text_data
        else:
            title_lower = clean_text(row.get('title', '')).lower()
            full_lower = title_lower + " " + clean_text(row.get('description', '')).lower()

        category_lower = category.lower() if category else ''

        predicted = None
        method = 'no_match'

        # Use pre-compiled matcher
        key = (category, feature_name)
        matcher = matchers.get(key)
        if matcher:
            predicted, method = matcher.match(full_lower, title_lower, category_lower)
        else:
            # Dynamic fallback for missing matchers
            allowed = tax_values.get(key, [])
            if allowed:
                cat_prior = feat_priors_cat.get(key, {})
                global_prior = feat_priors_global.get(feature_name, {})
                temp = CategoricalMatcher(allowed, feature_name, cat_prior, global_prior)
                predicted, method = temp.match(full_lower, title_lower, category_lower)
            elif feature_name in feat_priors_global:
                # No taxonomy entry — use global most common
                predicted = max(feat_priors_global[feature_name].items(), key=lambda x: x[1])[0]
                method = 'prior_global_notax'

        pred_str = str(predicted).strip() if predicted else ""
        is_correct = (pred_str == actual)

        total += 1
        method_stats[method or 'no_match'] += 1
        feature_stats[feature_name]['total'] += 1

        if is_correct:
            correct += 1
            method_correct[method or 'no_match'] += 1
            feature_stats[feature_name]['correct'] += 1
        else:
            if len(errors) < 500:
                errors.append({
                    'feature': feature_name,
                    'actual': actual,
                    'predicted': pred_str,
                    'method': method,
                    'category': category,
                    'title': title_lower[:120],
                })

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            acc = 100.0 * correct / total
            print(f"  [{i + 1:>7d}/{len(val_merged)}] acc={acc:.1f}% elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    accuracy = 100.0 * correct / total if total > 0 else 0

    # =============================================================================
    # RESULTS
    # =============================================================================
    print("\n" + "=" * 80)
    print("RESULTS - CATEGORICAL FEATURES ONLY")
    print("=" * 80)

    print(f"\n{'=' * 60}")
    print(f"  OVERALL CATEGORICAL ACCURACY: {correct}/{total} = {accuracy:.2f}%")
    print(f"  Time: {elapsed:.1f}s ({total / max(elapsed, 0.1):.0f} rows/sec)")
    print(f"{'=' * 60}")

    print(f"\nAccuracy by method:")
    print(f"  {'Method':<30s} | {'Correct':>8s} | {'Total':>8s} | {'Acc':>7s}")
    print(f"  {'-' * 30}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 7}")
    for meth in sorted(method_stats.keys(), key=lambda m: -method_stats[m]):
        mc = method_correct.get(meth, 0)
        mt = method_stats[meth]
        ma = 100 * mc / mt if mt > 0 else 0
        print(f"  {meth:<30s} | {mc:>8d} | {mt:>8d} | {ma:>6.1f}%")

    print(f"\nTop 30 features by volume:")
    feat_list = [(fn, s['correct'], s['total'],
                  100 * s['correct'] / s['total'] if s['total'] > 0 else 0)
                 for fn, s in feature_stats.items()]
    feat_list.sort(key=lambda x: -x[2])
    print(f"  {'Feature':<35s} | {'Corr':>7s} | {'Total':>7s} | {'Acc':>7s}")
    print(f"  {'-' * 35}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")
    for fname, fc, ft, fa in feat_list[:30]:
        print(f"  {fname:<35s} | {fc:>7d} | {ft:>7d} | {fa:>6.1f}%")

    print(f"\nWorst 20 features (>100 samples):")
    feat_low = [(f, c, t, a) for f, c, t, a in feat_list if t > 100]
    feat_low.sort(key=lambda x: x[3])
    for fname, fc, ft, fa in feat_low[:20]:
        print(f"  {fname:<35s} | {fc:>7d} | {ft:>7d} | {fa:>6.1f}%")

    print(f"\nSample errors (first 50):")
    for err in errors[:50]:
        m = err.get('method', 'no_match')
        print(f"  [{m:<25s}] {err['feature']:<25s}: pred='{err['predicted'][:40]:<40s}' actual='{err['actual'][:40]}'")

    print(f"\nTotal time: {time.time() - t_total:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"  TARGET: >60%  |  RESULT: {accuracy:.2f}% ({'PASSED' if accuracy >= 60 else 'NOT MET'})")
    print(f"{'=' * 60}")
