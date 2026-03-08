"""
Unified Feature Extractor — runs numeric and categorical extraction in parallel
on the full validation dataset, then reports per-type and combined accuracy.
"""

import pandas as pd
import numpy as np
import re
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATA LOADING (loaded once, used by both extractors)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("UNIFIED FEATURE EXTRACTOR")
print("=" * 80)
t_total = time.time()

products_train = pd.read_parquet("data/train/products.parquet")
features_train = pd.read_parquet("data/train/product_features.parquet")
products_val   = pd.read_parquet("data/val/products.parquet")
features_val   = pd.read_parquet("data/val/product_features.parquet")
taxonomy       = pd.read_parquet("data/taxonomy/taxonomy.parquet")

print(f"Train: {len(products_train)} products, {len(features_train)} features")
print(f"Val:   {len(products_val)} products, {len(features_val)} features")
print(f"Taxonomy: {len(taxonomy)} rows")

# --- Shuffled training data ------------------------------------------------
shuf_prod_path = "data/train/products.shuffled.parquet"
shuf_feat_path = "data/train/product_features.shuffled.parquet"
if os.path.exists(shuf_prod_path) and os.path.exists(shuf_feat_path):
    products_train = pd.read_parquet(shuf_prod_path)
    features_train = pd.read_parquet(shuf_feat_path)
else:
    products_train = products_train.sample(frac=1, random_state=42).reset_index(drop=True)
    features_train = features_train.sample(frac=1, random_state=42).reset_index(drop=True)
    products_train.to_parquet(shuf_prod_path)
    features_train.to_parquet(shuf_feat_path)
print(f"Shuffled train ready: {len(products_train)} products")

# --- Shared text pre-processing --------------------------------------------
_tag_re   = re.compile(r'<[^>]+>')
_space_re = re.compile(r'\s+')
_special_chars = re.compile(r'[®™©]')
_token_re = re.compile(
    r'[a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß0-9\-\+\.]*[a-zA-ZäöüÄÖÜß0-9]'
    r'|[a-zA-ZäöüÄÖÜß0-9]'
)

def clean_text(text):
    if not isinstance(text, str) or not text:
        return ""
    text = text.replace('<br>', ' ').replace('<br/>', ' ').replace('<br />', ' ')
    text = text.replace('<BR>', ' ').replace('<BR/>', ' ')
    text = _tag_re.sub(' ', text)
    text = _special_chars.sub(' ', text)
    text = _space_re.sub(' ', text)
    return text.strip()

print("Building shared text cache for validation products…")
t_clean = time.time()
uid_text_cache = {}
for _, row in products_val.iterrows():
    uid = row['uid']
    t_clean_val = clean_text(row['title']).lower()
    d_clean_val = clean_text(row['description']).lower()
    uid_text_cache[uid] = (t_clean_val, (t_clean_val + " " + d_clean_val).lower())
print(f"Cached {len(uid_text_cache)} products in {time.time() - t_clean:.1f}s")

def parse_aggregated_values(agg_str):
    """Parse taxonomy aggregated_feature_values string."""
    if not isinstance(agg_str, str) or not agg_str.strip():
        return []
    agg_str = agg_str.strip()
    if agg_str.startswith('{') and agg_str.endswith('}'):
        agg_str = agg_str[1:-1]
    values = []
    for m in re.finditer(r"'?\[([^\]]*)\]'?", agg_str):
        val = m.group(1).strip()
        if val:
            values.append(val)
    if not values:
        parts = agg_str.split("','")
        for p in parts:
            p = p.strip().strip("'").strip('[]').strip()
            if p:
                values.append(p)
    return values

print(f"Data loading done in {time.time() - t_total:.1f}s\n")


# █████████████████████████████████████████████████████████████████████████████
#  NUMERIC EXTRACTOR  (self-contained setup + evaluation function)
# █████████████████████████████████████████████████████████████████████████████

def build_numeric_extractor():
    """Build all numeric matchers / priors. Returns a callable evaluator."""
    print("[NUM] Building numeric extractor…")
    t0 = time.time()

    # --- Taxonomy ----------------------------------------------------------
    num_taxonomy = taxonomy[taxonomy['feature_type'] == 'numeric']
    num_tax_values = {}
    for _, row in num_taxonomy.iterrows():
        key = (row['category'], row['feature_name'])
        num_tax_values[key] = parse_aggregated_values(row['aggregated_feature_values'])

    # --- Training priors ---------------------------------------------------
    num_features_train = features_train[features_train['feature_type'] == 'numeric']
    train_merged = num_features_train.merge(
        products_train[['uid', 'category']], on='uid', how='left'
    )
    feat_priors = {}
    for fname, grp in train_merged.groupby('feature_name'):
        feat_priors[fname] = grp['feature_value'].value_counts().to_dict()
    feat_most_common = {fn: max(vc, key=vc.get) for fn, vc in feat_priors.items()}

    # --- Boundary helpers --------------------------------------------------
    _BEFORE_NUM_BOUNDARY = set('0123456789')
    _AFTER_NUM_BOUNDARY  = set('0123456789')

    def _find_bounded(haystack, needle, needle_starts_with_digit=False, needle_ends_with_digit=False):
        start = 0
        while True:
            idx = haystack.find(needle, start)
            if idx < 0:
                return -1
            ok = True
            if needle_starts_with_digit and idx > 0:
                ch = haystack[idx - 1]
                if ch in _BEFORE_NUM_BOUNDARY or ch == '.' or ch == ',':
                    ok = False
            if ok and needle_ends_with_digit:
                end = idx + len(needle)
                if end < len(haystack):
                    ch = haystack[end]
                    if ch in _AFTER_NUM_BOUNDARY or ch == '.' or ch == ',':
                        ok = False
            if ok:
                return idx
            start = idx + 1

    def _starts_digit(s):
        return len(s) > 0 and s[0] in '0123456789'

    def _ends_digit(s):
        return len(s) > 0 and s[-1] in '0123456789'

    # --- ValueMatcher class ------------------------------------------------
    class ValueMatcher:
        def __init__(self, allowed_values, feature_name, priors):
            self.allowed = allowed_values
            self.feature_name = feature_name
            self.priors = priors or {}
            self.allowed_sorted = sorted(allowed_values, key=lambda v: -len(v))
            self.allowed_lower_sorted = [
                (v, v.lower(), _starts_digit(v), _ends_digit(v))
                for v in self.allowed_sorted
            ]
            self.allowed_tight_sorted = sorted(
                [(v, v.lower().replace(' ', ''), _starts_digit(v), _ends_digit(v))
                 for v in allowed_values],
                key=lambda x: -len(x[1])
            )
            self.num_unit_pairs = []
            for v in allowed_values:
                m = re.match(r'^([\d\.,]+)\s+(.+)$', v)
                if m:
                    num_str = m.group(1)
                    unit = m.group(2).strip()
                    try:
                        num_f = float(num_str.replace(',', '.'))
                    except:
                        num_f = None
                    self.num_unit_pairs.append((v, num_str, unit, num_f))
            self.by_unit = defaultdict(list)
            for v, num_str, unit, num_f in self.num_unit_pairs:
                self.by_unit[unit.lower()].append((v, num_str, num_f))
            self.unit_regexes = {}
            for unit_lower in self.by_unit:
                unit_esc = re.escape(unit_lower)
                if len(unit_lower) <= 2 and unit_lower.isalpha():
                    pattern = r'(\d+(?:[.,]\d+)?)\s*' + unit_esc + r'(?:[²³\s,;.\-)/\]>]|$)'
                else:
                    pattern = r'(\d+(?:[.,]\d+)?)\s*' + unit_esc + r'(?:[\s,;.\-)/\]>]|$)'
                try:
                    self.unit_regexes[unit_lower] = re.compile(pattern, re.IGNORECASE)
                except:
                    pass
            self.prefix_vals = defaultdict(list)
            for v in allowed_values:
                m = re.match(r'^([A-Za-zÄÖÜäöü]+\.?\s*)(\d+(?:[.,]\d+)?)\s*(.*)$', v)
                if m:
                    prefix = m.group(1).strip().lower()
                    num = m.group(2)
                    try:
                        num_f = float(num.replace(',', '.'))
                    except:
                        num_f = None
                    self.prefix_vals[prefix].append((v, num, num_f))
            self.prefix_regexes = {}
            for prefix in self.prefix_vals:
                if prefix.endswith('.'):
                    p_esc = re.escape(prefix[:-1]) + r'\.?\s*'
                else:
                    p_esc = re.escape(prefix) + r'\.?\s*'
                p_esc += r'(\d+(?:[.,]\d+)?)'
                try:
                    self.prefix_regexes[prefix] = re.compile(p_esc, re.IGNORECASE)
                except:
                    pass
            self.special_patterns = []
            self._compile_specials()

        def _compile_specials(self):
            for v in self.allowed:
                if re.match(r'^:\d+$', v):
                    num = v[1:]
                    try:
                        pat = re.compile(r'(?:1\s*:\s*' + re.escape(num) + r'|:\s*' + re.escape(num) + r')(?:\b|\s|[,;)/]|$)')
                        self.special_patterns.append((pat, v))
                    except:
                        pass
                    continue
                m = re.match(r'^(\d+)°$', v)
                if m:
                    try:
                        pat = re.compile(r'(?<!\d)' + re.escape(m.group(1)) + r'\s*°')
                        self.special_patterns.append((pat, v))
                    except:
                        pass
                    continue
                m = re.match(r'^(\d+)-(\w+)$', v)
                if m:
                    try:
                        pat = re.compile(r'(?<!\d)' + re.escape(m.group(1)) + r'\s*-?\s*' + re.escape(m.group(2).lower()), re.IGNORECASE)
                        self.special_patterns.append((pat, v))
                    except:
                        pass
                    continue
                m = re.match(r'^ca\.\s*([\d\.,]+)\s+(.+)$', v)
                if m:
                    try:
                        pat = re.compile(r'(?:ca\.?\s*)?' + re.escape(m.group(1)) + r'\s*' + re.escape(m.group(2).lower()), re.IGNORECASE)
                        self.special_patterns.append((pat, v))
                    except:
                        pass
                    continue
                m = re.match(r'^ab\s+([\d\.,]+)\s+(.+)$', v)
                if m:
                    try:
                        pat = re.compile(r'(?:ab\s+)?' + re.escape(m.group(1)) + r'\s*' + re.escape(m.group(2).lower()), re.IGNORECASE)
                        self.special_patterns.append((pat, v))
                    except:
                        pass
                    continue
                m = re.match(r'^bis\s+([\d\.,]+)\s+(.+)$', v)
                if m:
                    try:
                        pat = re.compile(r'(?:bis\s+)?' + re.escape(m.group(1)) + r'\s*' + re.escape(m.group(2).lower()), re.IGNORECASE)
                        self.special_patterns.append((pat, v))
                    except:
                        pass
                    continue
                m = re.match(r'^(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s+(.+)$', v, re.IGNORECASE)
                if m:
                    try:
                        pat = re.compile(
                            r'(?<!\d)' + re.escape(m.group(1)) + r'\s*[xX×]\s*' + re.escape(m.group(2)) +
                            r'\s*[xX×]\s*' + re.escape(m.group(3)),
                            re.IGNORECASE
                        )
                        self.special_patterns.append((pat, v))
                    except:
                        pass
                    continue
                m = re.match(r'^(\d+)\s*x\s*(\d+)\s+(.+)$', v, re.IGNORECASE)
                if m:
                    try:
                        pat = re.compile(
                            r'(?<!\d)' + re.escape(m.group(1)) + r'\s*[xX×]\s*' + re.escape(m.group(2)) + r'\s*' + re.escape(m.group(3).lower()),
                            re.IGNORECASE
                        )
                        self.special_patterns.append((pat, v))
                        pat2 = re.compile(
                            r'(?<!\d)' + re.escape(m.group(1)) + r'\s*[xX×]\s*' + re.escape(m.group(2)) + r'(?:\s|[,;./)]|$)',
                            re.IGNORECASE
                        )
                        self.special_patterns.append((pat2, v))
                    except:
                        pass

        def _pick_best(self, candidates):
            if len(candidates) == 1:
                return candidates[0]
            return max(candidates, key=lambda v: self.priors.get(v, 0))

        def match(self, text_lower, title_lower):
            # STRATEGY 1: Bounded substring
            title_matches, desc_matches = [], []
            for val, val_lower, sd, ed in self.allowed_lower_sorted:
                idx = _find_bounded(title_lower, val_lower, sd, ed)
                if idx >= 0:
                    title_matches.append(val)
                else:
                    idx = _find_bounded(text_lower, val_lower, sd, ed)
                    if idx >= 0:
                        desc_matches.append(val)
            if title_matches:
                return self._pick_best(list(dict.fromkeys(title_matches))), 'exact_title'
            if desc_matches:
                return self._pick_best(list(dict.fromkeys(desc_matches))), 'exact_desc'

            # STRATEGY 2: Tight match
            text_tight = text_lower.replace(' ', '')
            title_tight = title_lower.replace(' ', '')
            for val, val_tight, sd, ed in self.allowed_tight_sorted:
                if len(val_tight) < 3:
                    continue
                for src in [title_tight, text_tight]:
                    idx = _find_bounded(src, val_tight, sd, ed)
                    if idx >= 0:
                        return val, 'tight'

            # STRATEGY 3: Special patterns
            sp = []
            for pat, val in self.special_patterns:
                if pat.search(text_lower):
                    sp.append(val)
            if sp:
                return self._pick_best(list(dict.fromkeys(sp))), 'special'

            # STRATEGY 4: Prefix match
            pf = []
            for prefix, vals in self.prefix_vals.items():
                regex = self.prefix_regexes.get(prefix)
                if regex is None:
                    continue
                for m in regex.finditer(text_lower):
                    found_num = m.group(1).replace(' ', '')
                    try:
                        found_f = float(found_num.replace(',', '.'))
                    except:
                        continue
                    for val, expected_num, expected_f in vals:
                        if found_num == expected_num:
                            pf.append(val)
                        elif expected_f is not None and abs(found_f - expected_f) < 0.01:
                            pf.append(val)
            if pf:
                return self._pick_best(list(dict.fromkeys(pf))), 'prefix'

            # STRATEGY 5: Number + unit regex
            nu_title, nu_desc = [], []
            for unit_lower, vals in self.by_unit.items():
                regex = self.unit_regexes.get(unit_lower)
                if regex is None:
                    continue
                for m in regex.finditer(text_lower):
                    found_num = m.group(1)
                    try:
                        found_f = float(found_num.replace(',', '.'))
                    except:
                        continue
                    in_title = m.start() < len(title_lower) + 3
                    for val, expected_num, expected_f in vals:
                        if expected_f is not None and abs(found_f - expected_f) < 0.01:
                            if in_title:
                                nu_title.append(val)
                            else:
                                nu_desc.append(val)
            if nu_title:
                return self._pick_best(list(dict.fromkeys(nu_title))), 'numunit_title'
            if nu_desc:
                return self._pick_best(list(dict.fromkeys(nu_desc))), 'numunit_desc'

            # STRATEGY 6: Dimension string parse
            dim_matches = []
            for m in re.finditer(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)', text_lower):
                for dim_str in [m.group(1), m.group(2), m.group(3)]:
                    try:
                        dim_f = float(dim_str.replace(',', '.'))
                    except:
                        continue
                    for val, num_str, unit, num_f in self.num_unit_pairs:
                        if num_f is not None and abs(dim_f - num_f) < 0.01:
                            dim_matches.append(val)
            if not dim_matches:
                for m in re.finditer(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)', text_lower):
                    for dim_str in [m.group(1), m.group(2)]:
                        try:
                            dim_f = float(dim_str.replace(',', '.'))
                        except:
                            continue
                        for val, num_str, unit, num_f in self.num_unit_pairs:
                            if num_f is not None and abs(dim_f - num_f) < 0.01:
                                dim_matches.append(val)
            if dim_matches:
                return self._pick_best(list(dict.fromkeys(dim_matches))), 'dimension'

            return None, None

    # --- Special handlers (module-level regexes) ----------------------------
    _dim3_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)')
    _dim2_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)')
    _hxbxt_re = re.compile(r'hxbxt\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
    _bxhxt_re = re.compile(r'bxhxt\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
    _bxtxh_re = re.compile(r'bxtxh\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
    _lxbxh_re = re.compile(r'lxbxh\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
    _lxb_re = re.compile(r'lxb\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
    _bxl_re = re.compile(r'bxl\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
    _stueck_re = re.compile(r'(?<!\d)(\d+)\s*(?:st\.?|stk\.?|stück|pcs)(?:\b|\s|$|[,;)/])', re.IGNORECASE)
    _ve_re = re.compile(r'VE\s*[=:]\s*(\d+)\s*(?:st\.?|stk\.?|stück|pcs)?', re.IGNORECASE)
    _faecher_mult_re = re.compile(r'(\d+)\s*[xX×]\s*(\d+)\s*fächer', re.IGNORECASE)
    _letter_re = re.compile(r'(?:buchstabe|zeichen|letter)\s*["\']?\s*([a-zäöüß])\s*["\']?', re.IGNORECASE)
    _letter_re2 = re.compile(r'["\']([a-zäöüß])["\']', re.IGNORECASE)
    _spann_range_re = re.compile(r'(?<!\d)(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)')
    _volstrom_re = re.compile(r'(\d+(?:[.,]?\d+)?)\s*(?:m³/h|cbm/h)', re.IGNORECASE)
    _sitzhoehe_re = re.compile(r'(?:sitz\w*höhe|sitzhöhe)\s*[:\s]*(\d+(?:[.,]\d+)?)\s*(mm|cm)?', re.IGNORECASE)

    def _parse_dim_num(s):
        try:
            return float(s.replace(',', '.'))
        except:
            return None

    # --- Special handler functions -----------------------------------------
    def handle_verpackungseinheit(text_lower, title_lower, allowed, priors):
        allowed_nums = {}
        for v in allowed:
            m = re.match(r'^(\d+)\s+Stück$', v)
            if m:
                allowed_nums[int(m.group(1))] = v
        if not allowed_nums:
            return None, None
        for m in _ve_re.finditer(text_lower):
            n = int(m.group(1))
            if n in allowed_nums:
                return allowed_nums[n], 'vpe_ve'
        for m in _stueck_re.finditer(title_lower):
            n = int(m.group(1))
            if n in allowed_nums:
                return allowed_nums[n], 'vpe_stk'
        for m in _stueck_re.finditer(text_lower):
            n = int(m.group(1))
            if n in allowed_nums:
                return allowed_nums[n], 'vpe_stk'
        return None, None

    def handle_einzelzeichen(text_lower, title_lower, allowed, priors):
        allowed_letters = {}
        for v in allowed:
            m = re.match(r'^([A-Za-zÄÖÜäöüß]+)\s*•?$', v.strip())
            if m:
                allowed_letters[m.group(1).upper()] = v
        if not allowed_letters:
            return None, None
        for pat in [_letter_re, _letter_re2]:
            m = pat.search(title_lower)
            if m:
                letter = m.group(1).upper()
                if letter in allowed_letters:
                    return allowed_letters[letter], 'einzelzeichen'
        for m in re.finditer(r'\.(\w)/|\.\d+\.([A-Z])(?:\b|/)', text_lower, re.IGNORECASE):
            letter = (m.group(1) or m.group(2) or '').upper()
            if letter in allowed_letters:
                return allowed_letters[letter], 'einzelzeichen'
        for m in re.finditer(r'\d{4,5}\.\d{3}\.([a-z])', title_lower, re.IGNORECASE):
            letter = m.group(1).upper()
            if letter in allowed_letters:
                return allowed_letters[letter], 'einzelzeichen'
        return None, None

    def handle_faecher(text_lower, title_lower, allowed, priors):
        allowed_nums = {}
        for v in allowed:
            m = re.match(r'^(\d+)\s+Fächer$', v)
            if m:
                allowed_nums[int(m.group(1))] = v
        if not allowed_nums:
            return None, None
        for m in _faecher_mult_re.finditer(text_lower):
            product = int(m.group(1)) * int(m.group(2))
            if product in allowed_nums:
                return allowed_nums[product], 'faecher_mult'
        for m in re.finditer(r'(\d+)\s*fächer', text_lower, re.IGNORECASE):
            n = int(m.group(1))
            if n in allowed_nums:
                return allowed_nums[n], 'faecher_direct'
        return None, None

    def handle_luftdurchsatz(text_lower, title_lower, allowed, priors):
        ab_vals = []
        for v in allowed:
            m = re.match(r'^ab\s+([\d\.,]+)\s+m³/h$', v)
            if m:
                try:
                    num = float(m.group(1).replace(',', '.'))
                    ab_vals.append((v, num))
                except:
                    pass
        if not ab_vals:
            return None, None
        ab_vals.sort(key=lambda x: x[1])
        for m in _volstrom_re.finditer(text_lower):
            try:
                found = float(m.group(1).replace(',', '.'))
            except:
                continue
            best, best_diff = None, float('inf')
            for val, num in ab_vals:
                diff = abs(found - num)
                if diff < best_diff:
                    best_diff = diff
                    best = val
            if best and best_diff <= found * 0.15:
                return best, 'luft_volstrom'
        return None, None

    def handle_sitzhoehe(text_lower, title_lower, allowed, priors):
        allowed_nums = {}
        for v in allowed:
            m = re.match(r'^([\d\.,]+)\s+(.+)$', v)
            if m:
                try:
                    n = float(m.group(1).replace(',', '.'))
                    allowed_nums[n] = v
                except:
                    pass
        if not allowed_nums:
            return None, None
        m = _sitzhoehe_re.search(text_lower)
        if m:
            try:
                n = float(m.group(1).replace(',', '.'))
                unit = m.group(2) or 'mm'
                if unit == 'cm':
                    n *= 10
                if n in allowed_nums:
                    return allowed_nums[n], 'sitzhoehe'
            except:
                pass
        return None, None

    def handle_spannbereich(text_lower, title_lower, allowed, feat_name):
        is_von = 'von' in feat_name.lower() or 'min' in feat_name.lower()
        allowed_nums = {}
        for v in allowed:
            m = re.match(r'^([\d\.,]+)\s+(.+)$', v)
            if m:
                try:
                    num = float(m.group(1).replace(',', '.'))
                    allowed_nums[num] = v
                except:
                    pass
        if not allowed_nums:
            return None, None
        for m in _spann_range_re.finditer(text_lower):
            try:
                n1 = float(m.group(1).replace(',', '.'))
                n2 = float(m.group(2).replace(',', '.'))
            except:
                continue
            target = n1 if is_von else n2
            if target in allowed_nums:
                return allowed_nums[target], 'spann_range'
            for allowed_n, v in allowed_nums.items():
                if abs(target - allowed_n) < 0.1:
                    return v, 'spann_range'
        return None, None

    def handle_dimension_by_name(text_lower, title_lower, feature_name, allowed, priors):
        fname_lower = feature_name.lower()
        allowed_nums = {}
        for v in allowed:
            m = re.match(r'^([\d\.,]+)\s+(.+)$', v)
            if m:
                try:
                    num = float(m.group(1).replace(',', '.'))
                    allowed_nums[num] = v
                except:
                    pass
        if not allowed_nums:
            return None, None
        dim_maps = [
            (_hxbxt_re, {'höhe': 1, 'gesamt_höhe': 1, 'gesamthöhe': 1, 'breite': 2, 'gesamt_breite': 2, 'gesamtbreite': 2, 'tiefe': 3, 'gesamt_tiefe': 3, 'gesamttiefe': 3}),
            (_bxhxt_re, {'breite': 1, 'gesamtbreite': 1, 'höhe': 2, 'gesamthöhe': 2, 'tiefe': 3, 'gesamttiefe': 3}),
            (_bxtxh_re, {'breite': 1, 'gesamtbreite': 1, 'tiefe': 2, 'gesamttiefe': 2, 'höhe': 3, 'gesamthöhe': 3}),
            (_lxbxh_re, {'länge': 1, 'gesamtlänge': 1, 'breite': 2, 'gesamtbreite': 2, 'höhe': 3, 'gesamthöhe': 3}),
            (_lxb_re,   {'länge': 1, 'gesamtlänge': 1, 'breite': 2, 'gesamtbreite': 2}),
            (_bxl_re,   {'breite': 1, 'gesamtbreite': 1, 'länge': 2, 'gesamtlänge': 2}),
        ]
        fname_clean = fname_lower.replace('-', '_').replace(' ', '_')
        for regex, mapping in dim_maps:
            m = regex.search(text_lower)
            if m:
                for key, grp_idx in mapping.items():
                    if key in fname_clean:
                        try:
                            num = float(m.group(grp_idx).replace(',', '.'))
                            if num in allowed_nums:
                                return allowed_nums[num], 'dim_labeled'
                        except:
                            pass
        return None, None

    def handle_dim_position(text_lower, title_lower, feature_name, allowed, category):
        allowed_nums = {}
        for v in allowed:
            m = re.match(r'^([\d\.,]+)\s+(.+)$', v)
            if m:
                n = _parse_dim_num(m.group(1))
                if n is not None:
                    allowed_nums[n] = v
        if not allowed_nums:
            return None, None
        fname_l = feature_name.lower()
        pos_map = {
            'ø unten': 0, 'unten': 0,
            'ø oben': 1, 'oben': 1,
            'höhe': 2, 'kopfhöhe': 2,
            'innen-ø': 0, 'innendurchmesser': 0,
            'außen-ø': 1, 'außendurchmesser': 1,
            'kopf-ø': 0,
        }
        target_pos = None
        for key, pos in pos_map.items():
            if key == fname_l or fname_l.startswith(key):
                target_pos = pos
                break
        if target_pos is None:
            return None, None
        for m in _dim3_re.finditer(text_lower):
            dims = [m.group(1), m.group(2), m.group(3)]
            if target_pos < len(dims):
                n = _parse_dim_num(dims[target_pos])
                if n in allowed_nums:
                    return allowed_nums[n], 'dim_position'
        for m in _dim3_re.finditer(title_lower):
            dims = [m.group(1), m.group(2), m.group(3)]
            if target_pos < len(dims):
                n = _parse_dim_num(dims[target_pos])
                if n in allowed_nums:
                    return allowed_nums[n], 'dim_position'
        if fname_l in ('innen-ø', 'außen-ø'):
            washer_pat = re.compile(r'(?:scheibe|washer)\w*\s+([\d\.,]+)\s*mm\s+([\d\.,]+)\s*mm', re.IGNORECASE)
            for m in washer_pat.finditer(text_lower):
                pos = 0 if fname_l.startswith('innen') else 1
                n = _parse_dim_num(m.group(pos + 1))
                if n in allowed_nums:
                    return allowed_nums[n], 'dim_washer'
            iso_pat = re.compile(r'iso\d+\s+([\d\.,]+)\s*[xX×]\s*([\d\.,]+)\s*[xX×]\s*([\d\.,]+)', re.IGNORECASE)
            for m in iso_pat.finditer(text_lower):
                pos = 0 if fname_l.startswith('innen') else 1
                n = _parse_dim_num(m.group(pos + 1))
                if n in allowed_nums:
                    return allowed_nums[n], 'dim_washer'
        if fname_l == 'kopf-ø' and target_pos == 0:
            for m in _dim2_re.finditer(title_lower):
                n = _parse_dim_num(m.group(1))
                if n in allowed_nums:
                    return allowed_nums[n], 'dim_position'
        return None, None

    def handle_range_feature(text_lower, title_lower, allowed, feat_name):
        is_min = any(k in feat_name.lower() for k in ['von', 'min'])
        is_max = any(k in feat_name.lower() for k in ['bis', 'max'])
        allowed_nums = defaultdict(list)
        for v in allowed:
            m = re.match(r'^([\d\.,]+)\s+(.+)$', v)
            if m:
                n = _parse_dim_num(m.group(1))
                if n is not None:
                    allowed_nums[n].append(v)
        if not allowed_nums:
            return None, None
        def _pick_fmt(vals, raw):
            if len(vals) == 1:
                return vals[0]
            if '.' not in raw:
                for v in vals:
                    m = re.match(r'^(\d+)\s', v)
                    if m:
                        return v
            else:
                for v in vals:
                    if '.' in v.split()[0] or ',' in v.split()[0]:
                        return v
            return vals[0]
        range_pats = [
            re.compile(r'(?<!\d)([\d\.,]+)\s*[-–—]\s*([\d\.,]+)\s*(mm|cm|m|µm|µl|ml|l|kg|g|bar|°C|A|V|W|kW|kN|N|nm|µm|g/ml|g/cm³|g/cm3)', re.IGNORECASE),
            re.compile(r'(?<!\d)([\d\.,]+)\s+bis\s+([\d\.,]+)\s*(mm|cm|m|µm|µl|ml|l|kg|g|bar|°C|A|V|W|kW|kN|N|nm|µm|g/ml|g/cm³|g/cm3)', re.IGNORECASE),
            re.compile(r'(?<!\d)([\d\.,]+)\s*\.{2,3}\s*([\d\.,]+)', re.IGNORECASE),
            re.compile(r'(?<!\d)([\d\.,]+)\s*[-–—]\s*([\d\.,]+)\s*:\s*[\d\.,]+', re.IGNORECASE),
            re.compile(r'(?<!\d)([\d\.,]+)\s*[-–—]\s*([\d\.,]+)(?=\s|$|[,;)/])', re.IGNORECASE),
        ]
        for pat in range_pats:
            for m in pat.finditer(text_lower):
                n1, n2 = _parse_dim_num(m.group(1)), _parse_dim_num(m.group(2))
                if n1 is None or n2 is None:
                    continue
                raw1, raw2 = m.group(1), m.group(2)
                if is_max:
                    target, raw = n2, raw2
                elif is_min:
                    target, raw = n1, raw1
                else:
                    target, raw = n1, raw1
                if target in allowed_nums:
                    return _pick_fmt(allowed_nums[target], raw), 'range_parse'
                for an, avs in allowed_nums.items():
                    if abs(target - an) < 0.1:
                        return _pick_fmt(avs, raw), 'range_parse'
        return None, None

    # --- Feature name sets for special handlers ----------------------------
    SPANN_FEATURES = {
        'Spannbereich von', 'Spannbereich bis',
        'min. Spannbereich', 'max. Spannbereich',
        'Schweißbereich min.', 'Schweißbereich max.',
        'Schneidbereich min.', 'Schneidbereich max.',
        'Messbereich von', 'Messbereich bis',
        'min. Bohr-Ø', 'max. Bohr-Ø',
        'Klemmbereich', 'Klemmbereich von', 'Klemmbereich bis',
        'min. Durchmesser', 'max. Durchmesser',
        'Min. Volumen', 'Max. Volumen',
        'Beschleunigung von', 'Beschleunigung bis',
        'Geschwindigkeit', 'Drehzahl von', 'Drehzahl bis',
        'Wegeanzahl',
    }
    DIM_FEATURES = {
        'Gesamtbreite', 'Gesamthöhe', 'Gesamttiefe', 'Gesamtlänge',
        'Breite', 'Höhe', 'Tiefe', 'Länge',
        'Sitzhöhe', 'Kopfhöhe',
    }
    SPECIAL_FIRST_FEATURES = {
        'Verpackungseinheit', 'Einzelzeichen', 'Fächeranzahl',
        'Luftdurchsatz', 'Sitzhöhe',
        'Ø unten', 'Ø oben', 'Höhe',
        'Innen-Ø', 'Außen-Ø', 'Kopf-Ø',
        'Messbereich von', 'Messbereich bis',
        'Min. Volumen', 'Max. Volumen',
    } | SPANN_FEATURES | DIM_FEATURES

    # --- Compile all numeric matchers --------------------------------------
    num_matchers = {}
    for (cat, fname), vals in num_tax_values.items():
        if vals:
            priors = feat_priors.get(fname, {})
            num_matchers[(cat, fname)] = ValueMatcher(vals, fname, priors)

    print(f"[NUM] {len(num_matchers)} matchers compiled in {time.time() - t0:.1f}s")

    # --- Evaluation function (returned) ------------------------------------
    def evaluate_numeric():
        """Run numeric evaluation on the full val set. Returns dict with results."""
        num_features_v = features_val[features_val['feature_type'] == 'numeric'].copy()
        val_merged = num_features_v.merge(
            products_val[['uid', 'category']], on='uid', how='left'
        )
        correct = 0
        total = 0
        method_stats = Counter()
        method_correct = Counter()
        feature_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        errors = []
        t0_inner = time.time()

        for i, (_, row) in enumerate(val_merged.iterrows()):
            uid = row['uid']
            category = row['category']
            feature_name = row['feature_name']
            actual = str(row['feature_value']).strip()
            predicted = None
            method = 'no_match'

            text_data = uid_text_cache.get(uid)
            if text_data is None:
                title_lower = ""
                full_lower = ""
            else:
                title_lower, full_lower = text_data

            key = (category, feature_name)
            allowed = num_tax_values.get(key, [])
            priors_for_feat = feat_priors.get(feature_name, {})

            # Special handlers first
            if allowed and feature_name in SPECIAL_FIRST_FEATURES:
                if feature_name == 'Verpackungseinheit':
                    predicted, method = handle_verpackungseinheit(full_lower, title_lower, allowed, priors_for_feat)
                elif feature_name == 'Einzelzeichen':
                    predicted, method = handle_einzelzeichen(full_lower, title_lower, allowed, priors_for_feat)
                elif feature_name == 'Fächeranzahl':
                    predicted, method = handle_faecher(full_lower, title_lower, allowed, priors_for_feat)
                elif feature_name == 'Luftdurchsatz':
                    predicted, method = handle_luftdurchsatz(full_lower, title_lower, allowed, priors_for_feat)
                elif feature_name == 'Sitzhöhe':
                    predicted, method = handle_sitzhoehe(full_lower, title_lower, allowed, priors_for_feat)
                elif feature_name in ('Ø unten', 'Ø oben', 'Höhe'):
                    predicted, method = handle_dim_position(full_lower, title_lower, feature_name, allowed, category)
                    if predicted is None and feature_name in DIM_FEATURES:
                        predicted, method = handle_dimension_by_name(full_lower, title_lower, feature_name, allowed, priors_for_feat)
                elif feature_name in ('Innen-Ø', 'Außen-Ø', 'Kopf-Ø'):
                    predicted, method = handle_dim_position(full_lower, title_lower, feature_name, allowed, category)
                elif feature_name in SPANN_FEATURES or feature_name in ('Messbereich von', 'Messbereich bis', 'Min. Volumen', 'Max. Volumen'):
                    predicted, method = handle_range_feature(full_lower, title_lower, allowed, feature_name)
                elif feature_name in DIM_FEATURES:
                    predicted, method = handle_dimension_by_name(full_lower, title_lower, feature_name, allowed, priors_for_feat)
                    if predicted is None:
                        predicted, method = handle_range_feature(full_lower, title_lower, allowed, feature_name)

            # Main matcher
            if predicted is None:
                matcher = num_matchers.get(key)
                if matcher:
                    predicted, method = matcher.match(full_lower, title_lower)

            # Remaining special
            if predicted is None and allowed and feature_name not in SPECIAL_FIRST_FEATURES:
                fname_l = feature_name.lower()
                if any(k in fname_l for k in ['von', 'bis', 'min', 'max', 'bereich']):
                    predicted, method = handle_range_feature(full_lower, title_lower, allowed, feature_name)

            # Fallbacks
            if predicted is None:
                if len(allowed) == 1:
                    predicted = allowed[0]
                    method = 'single_value'
            if predicted is None and allowed:
                cat_clean = category.lower().replace('_', '').replace('-', '')
                for val in allowed:
                    if val.lower().replace(' ', '') in cat_clean:
                        predicted = val
                        method = 'cat_derived'
                        break
            if predicted is None:
                if feature_name in feat_priors:
                    if allowed:
                        allowed_set = set(allowed)
                        for val, cnt in sorted(feat_priors[feature_name].items(), key=lambda x: -x[1]):
                            if val in allowed_set:
                                predicted = val
                                method = 'prior_allowed'
                                break
                    if predicted is None:
                        predicted = feat_most_common.get(feature_name)
                        method = 'prior_global'
            if predicted is None and feature_name in feat_most_common:
                predicted = feat_most_common[feature_name]
                method = 'prior_global'

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
                if len(errors) < 200:
                    errors.append({
                        'feature': feature_name,
                        'actual': actual,
                        'predicted': pred_str,
                        'method': method or 'no_match',
                    })
            if (i + 1) % 100000 == 0:
                elapsed = time.time() - t0_inner
                acc = 100 * correct / total
                print(f"  [NUM {i+1:>7d}/{len(val_merged)}] acc={acc:.1f}% elapsed={elapsed:.0f}s")

        elapsed = time.time() - t0_inner
        accuracy = 100 * correct / total if total > 0 else 0
        return {
            'type': 'numeric',
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'elapsed': elapsed,
            'method_stats': dict(method_stats),
            'method_correct': dict(method_correct),
            'feature_stats': dict(feature_stats),
            'errors': errors,
        }

    return evaluate_numeric


# █████████████████████████████████████████████████████████████████████████████
#  CATEGORICAL EXTRACTOR  (self-contained setup + evaluation function)
# █████████████████████████████████████████████████████████████████████████████

def build_categorical_extractor():
    """Build all categorical matchers / priors. Returns a callable evaluator."""
    print("[CAT] Building categorical extractor…")
    t0 = time.time()

    # --- Taxonomy ----------------------------------------------------------
    cat_taxonomy = taxonomy[taxonomy['feature_type'] == 'categorical']
    cat_tax_values = {}
    for _, row in cat_taxonomy.iterrows():
        key = (row['category'], row['feature_name'])
        cat_tax_values[key] = parse_aggregated_values(row['aggregated_feature_values'])

    # --- Training priors ---------------------------------------------------
    cat_features_train = features_train[features_train['feature_type'] == 'categorical']
    train_merged = cat_features_train.merge(
        products_train[['uid', 'category']], on='uid', how='left'
    )
    feat_priors_cat = {}
    for (cat, fname), grp in train_merged.groupby(['category', 'feature_name']):
        feat_priors_cat[(cat, fname)] = grp['feature_value'].value_counts().to_dict()
    feat_priors_global = {}
    for fname, grp in train_merged.groupby('feature_name'):
        feat_priors_global[fname] = grp['feature_value'].value_counts().to_dict()

    # --- Keyword scores ----------------------------------------------------
    print("[CAT] Building keyword scores…")
    train_with_text = train_merged.merge(
        products_train[['uid', 'title']], on='uid', how='left'
    )
    keyword_scores_by_feature = {}
    for fname, grp in train_with_text.groupby('feature_name'):
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
        value_keywords = {}
        for val, tcounts in value_token_counts.items():
            n_docs_val = value_total_docs[val]
            if n_docs_val < 3:
                continue
            keywords = {}
            for t, count in tcounts.items():
                if count < 3:
                    continue
                specificity = count / total_token_counts[t]
                frequency = count / n_docs_val
                if specificity > 0.3 and frequency > 0.05:
                    keywords[t] = specificity * frequency
            if keywords:
                value_keywords[val] = dict(sorted(keywords.items(), key=lambda x: -x[1])[:30])
        if value_keywords:
            keyword_scores_by_feature[fname] = value_keywords

    # --- CategoricalMatcher class ------------------------------------------
    class CategoricalMatcher:
        def __init__(self, allowed_values, feature_name, cat_priors, global_priors, keyword_scores=None):
            self.allowed = allowed_values
            self.feature_name = feature_name
            self.cat_priors = cat_priors or {}
            self.global_priors = global_priors or {}
            self.keyword_scores = keyword_scores or {}
            self.allowed_sorted = sorted(allowed_values, key=len, reverse=True)
            self.boundary_patterns = []
            for v in self.allowed_sorted:
                vl = v.lower()
                try:
                    pat = re.compile(
                        r'(?<![a-zA-Z0-9äöüÄÖÜß])' + re.escape(vl) + r'(?![a-zA-Z0-9äöüÄÖÜß])',
                        re.IGNORECASE
                    )
                    self.boundary_patterns.append((v, vl, pat))
                except re.error:
                    self.boundary_patterns.append((v, vl, None))
            self.tight_forms = [(v, re.sub(r'[\s\-_]', '', v.lower())) for v in self.allowed_sorted if len(re.sub(r'[\s\-_]', '', v.lower())) >= 2]
            self.umlaut_forms = []
            for v in self.allowed_sorted:
                vl = v.lower()
                norm = vl.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
                if norm != vl:
                    self.umlaut_forms.append((v, norm))
                rev = vl.replace('ae', 'ä').replace('oe', 'ö').replace('ue', 'ü').replace('ss', 'ß')
                if rev != vl:
                    self.umlaut_forms.append((v, rev))
            self.token_sets = []
            for v in self.allowed_sorted:
                tokens = set(re.findall(r'[a-zA-ZäöüÄÖÜß0-9]+', v.lower()))
                if len(tokens) >= 2 and all(len(t) >= 2 for t in tokens):
                    self.token_sets.append((v, tokens))
            self.paren_forms = []
            for v in allowed_values:
                m = re.match(r'^(.+?)\s*\((.+?)\)$', v)
                if m:
                    main = m.group(1).strip().lower()
                    qual = m.group(2).strip().lower()
                    if len(main) >= 2 and len(qual) >= 1:
                        self.paren_forms.append((v, main, qual))
            self.slash_forms = []
            for v in allowed_values:
                if '/' in v:
                    parts = [p.strip().lower() for p in v.split('/') if len(p.strip()) >= 2]
                    if len(parts) >= 2:
                        self.slash_forms.append((v, parts))
            self.paren_extensions = defaultdict(list)
            for v, main, qual in self.paren_forms:
                self.paren_extensions[main].append((v, qual))
            self.longer_values = defaultdict(list)
            for v in allowed_values:
                vl = v.lower()
                for v2 in allowed_values:
                    v2l = v2.lower()
                    if v2l != vl and vl in v2l and len(v2l) > len(vl):
                        self.longer_values[vl].append(v2)

        def _pick_best(self, candidates, title_lower=None):
            if len(candidates) == 1:
                return candidates[0]
            candidates_lower = [(c, c.lower()) for c in candidates]
            non_contained = []
            for c, cl in candidates_lower:
                is_contained = any(c2l != cl and cl in c2l and len(c2l) > len(cl) for _, c2l in candidates_lower)
                if not is_contained:
                    non_contained.append(c)
            if len(non_contained) == 1:
                return non_contained[0]
            if not non_contained:
                non_contained = candidates
            if self.cat_priors:
                best = max(non_contained, key=lambda c: self.cat_priors.get(c, 0))
                if self.cat_priors.get(best, 0) > 0:
                    return best
            if self.global_priors:
                best = max(non_contained, key=lambda c: self.global_priors.get(c, 0))
                if self.global_priors.get(best, 0) > 0:
                    return best
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
            return max(non_contained, key=len)

        def _refine_with_paren(self, base_val, title_lower, full_text_lower):
            base_lower = base_val.lower()
            extensions = self.paren_extensions.get(base_lower, [])
            if extensions:
                title_matches, full_matches = [], []
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
            longer_vals = self.longer_values.get(base_lower, [])
            if longer_vals:
                for lv in longer_vals:
                    lvl = lv.lower()
                    try:
                        pat = re.compile(r'(?<![a-zA-Z0-9äöüÄÖÜß])' + re.escape(lvl) + r'(?![a-zA-Z0-9äöüÄÖÜß])', re.IGNORECASE)
                        if pat.search(title_lower):
                            return lv
                        if pat.search(full_text_lower):
                            return lv
                    except re.error:
                        pass
                tight_title = re.sub(r'[\s\-_]', '', title_lower)
                tight_full = re.sub(r'[\s\-_]', '', full_text_lower)
                for lv in longer_vals:
                    tight_lv = re.sub(r'[\s\-_]', '', lv.lower())
                    if len(tight_lv) >= 5:
                        if tight_lv in tight_title or tight_lv in tight_full:
                            return lv
            return base_val

        def _best_prior(self, allowed_set=None):
            if self.cat_priors:
                for val, _ in sorted(self.cat_priors.items(), key=lambda x: -x[1]):
                    if allowed_set is None or val in allowed_set:
                        return val, 'prior_cat'
            if self.global_priors:
                for val, _ in sorted(self.global_priors.items(), key=lambda x: -x[1]):
                    if allowed_set is None or val in allowed_set:
                        return val, 'prior_global'
            return None, None

        def match(self, full_text_lower, title_lower, category_lower):
            if len(self.allowed) == 1:
                return self.allowed[0], 'single_value'
            title_hits, desc_hits = [], []
            for v, vl, pat in self.boundary_patterns:
                if pat is not None:
                    if pat.search(title_lower):
                        title_hits.append(v)
                    elif pat.search(full_text_lower):
                        desc_hits.append(v)
            if title_hits:
                best = self._pick_best(list(dict.fromkeys(title_hits)), title_lower)
                return self._refine_with_paren(best, title_lower, full_text_lower), 'boundary_title'
            if desc_hits:
                best = self._pick_best(list(dict.fromkeys(desc_hits)), title_lower)
                return self._refine_with_paren(best, title_lower, full_text_lower), 'boundary_desc'
            paren_title, paren_desc = [], []
            for v, main, qual in self.paren_forms:
                if main in title_lower and qual in title_lower:
                    paren_title.append(v)
                elif main in full_text_lower and qual in full_text_lower:
                    paren_desc.append(v)
            if paren_title:
                return self._pick_best(list(dict.fromkeys(paren_title))), 'paren_title'
            if paren_desc:
                return self._pick_best(list(dict.fromkeys(paren_desc))), 'paren_desc'
            title_sub, desc_sub = [], []
            for v in self.allowed_sorted:
                vl = v.lower()
                if len(vl) >= 3:
                    if vl in title_lower:
                        title_sub.append(v)
                    elif vl in full_text_lower:
                        desc_sub.append(v)
            if title_sub:
                best = self._pick_best(list(dict.fromkeys(title_sub)), title_lower)
                return self._refine_with_paren(best, title_lower, full_text_lower), 'substr_title'
            if desc_sub:
                best = self._pick_best(list(dict.fromkeys(desc_sub)), title_lower)
                return self._refine_with_paren(best, title_lower, full_text_lower), 'substr_desc'
            tight_title = re.sub(r'[\s\-_]', '', title_lower)
            tight_full = re.sub(r'[\s\-_]', '', full_text_lower)
            tight_t, tight_d = [], []
            for v, tight in self.tight_forms:
                if len(tight) >= 3:
                    if tight in tight_title:
                        tight_t.append(v)
                    elif tight in tight_full:
                        tight_d.append(v)
            if tight_t:
                return self._pick_best(list(dict.fromkeys(tight_t))), 'tight_title'
            if tight_d:
                return self._pick_best(list(dict.fromkeys(tight_d))), 'tight_desc'
            for v, norm in self.umlaut_forms:
                if norm in title_lower:
                    return v, 'umlaut_title'
                if norm in full_text_lower:
                    return v, 'umlaut_desc'
            title_tokens = set(_token_re.findall(title_lower))
            full_tokens = set(_token_re.findall(full_text_lower))
            token_title, token_desc = [], []
            for v, tokens in self.token_sets:
                if tokens.issubset(title_tokens):
                    token_title.append(v)
                elif tokens.issubset(full_tokens):
                    token_desc.append(v)
            if token_title:
                return self._pick_best(list(dict.fromkeys(token_title))), 'token_title'
            if token_desc:
                return self._pick_best(list(dict.fromkeys(token_desc))), 'token_desc'
            for v, parts in self.slash_forms:
                if all(p in title_lower for p in parts):
                    return v, 'slash_title'
                if all(p in full_text_lower for p in parts):
                    return v, 'slash_desc'
            if self.keyword_scores:
                best_val, best_score = None, 0.0
                for v in self.allowed:
                    score = 0.0
                    v_keywords = self.keyword_scores.get(v)
                    if v_keywords:
                        for tok, s in v_keywords.items():
                            if tok in title_tokens:
                                score += s * 2.0
                            elif tok in full_tokens:
                                score += s
                    if score > best_score:
                        best_score = score
                        best_val = v
                if best_val and best_score > 0.2:
                    return best_val, 'keyword_vote'
            allowed_set = set(self.allowed)
            prior_val, prior_method = self._best_prior(allowed_set)
            if prior_val:
                return prior_val, prior_method
            if self.allowed:
                return self.allowed[0], 'fallback_first'
            return None, None

    # --- Compile all categorical matchers ----------------------------------
    cat_matchers = {}
    for (cat, fname), vals in cat_tax_values.items():
        if vals:
            cat_prior = feat_priors_cat.get((cat, fname), {})
            global_prior = feat_priors_global.get(fname, {})
            kw_scores = keyword_scores_by_feature.get(fname, {})
            cat_matchers[(cat, fname)] = CategoricalMatcher(vals, fname, cat_prior, global_prior, kw_scores)

    print(f"[CAT] {len(cat_matchers)} matchers compiled in {time.time() - t0:.1f}s")

    # --- Evaluation function (returned) ------------------------------------
    def evaluate_categorical():
        """Run categorical evaluation on the full val set. Returns dict with results."""
        cat_features_v = features_val[features_val['feature_type'] == 'categorical'].copy()
        val_merged = cat_features_v.merge(
            products_val[['uid', 'category']], on='uid', how='left'
        )
        correct = 0
        total = 0
        method_stats = Counter()
        method_correct = Counter()
        feature_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        errors = []
        t0_inner = time.time()

        for i, (_, row) in enumerate(val_merged.iterrows()):
            uid = row['uid']
            category = row['category']
            feature_name = row['feature_name']
            actual = str(row['feature_value']).strip()

            text_data = uid_text_cache.get(uid)
            if text_data is None:
                title_lower = ""
                full_lower = ""
            else:
                title_lower, full_lower = text_data

            category_lower = category.lower() if category else ''
            predicted = None
            method = 'no_match'

            key = (category, feature_name)
            matcher = cat_matchers.get(key)
            if matcher:
                predicted, method = matcher.match(full_lower, title_lower, category_lower)
            else:
                allowed = cat_tax_values.get(key, [])
                if allowed:
                    cat_prior = feat_priors_cat.get(key, {})
                    global_prior = feat_priors_global.get(feature_name, {})
                    temp = CategoricalMatcher(allowed, feature_name, cat_prior, global_prior)
                    predicted, method = temp.match(full_lower, title_lower, category_lower)
                elif feature_name in feat_priors_global:
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
                if len(errors) < 200:
                    errors.append({
                        'feature': feature_name,
                        'actual': actual,
                        'predicted': pred_str,
                        'method': method or 'no_match',
                    })
            if (i + 1) % 100000 == 0:
                elapsed = time.time() - t0_inner
                acc = 100 * correct / total
                print(f"  [CAT {i+1:>7d}/{len(val_merged)}] acc={acc:.1f}% elapsed={elapsed:.0f}s")

        elapsed = time.time() - t0_inner
        accuracy = 100 * correct / total if total > 0 else 0
        return {
            'type': 'categorical',
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'elapsed': elapsed,
            'method_stats': dict(method_stats),
            'method_correct': dict(method_correct),
            'feature_stats': dict(feature_stats),
            'errors': errors,
        }

    return evaluate_categorical


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD BOTH EXTRACTORS (sequential — compilation / prior building)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("BUILDING EXTRACTORS")
print("=" * 80)

eval_numeric      = build_numeric_extractor()
eval_categorical  = build_categorical_extractor()


# ═══════════════════════════════════════════════════════════════════════════════
# RUN EVALUATIONS IN PARALLEL (ThreadPoolExecutor — two threads)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("RUNNING PARALLEL EVALUATION ON FULL VALIDATION SET")
print("=" * 80)

t_eval = time.time()

with ThreadPoolExecutor(max_workers=2) as executor:
    future_num = executor.submit(eval_numeric)
    future_cat = executor.submit(eval_categorical)

    results_num = future_num.result()
    results_cat = future_cat.result()

eval_elapsed = time.time() - t_eval


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

combined_correct = results_num['correct'] + results_cat['correct']
combined_total   = results_num['total']   + results_cat['total']
combined_accuracy = 100.0 * combined_correct / combined_total if combined_total > 0 else 0

print("\n" + "=" * 80)
print("RESULTS — NUMERIC FEATURES")
print("=" * 80)
rn = results_num
print(f"\n  Accuracy: {rn['correct']}/{rn['total']} = {rn['accuracy']:.2f}%")
print(f"  Time:     {rn['elapsed']:.1f}s ({rn['total']/max(rn['elapsed'],0.1):.0f} rows/sec)")
print(f"\n  Method breakdown:")
print(f"  {'Method':<25s} | {'Correct':>8s} | {'Total':>8s} | {'Acc':>7s}")
print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
for meth in sorted(rn['method_stats'].keys(), key=lambda m: -rn['method_stats'][m]):
    mc = rn['method_correct'].get(meth, 0)
    mt = rn['method_stats'][meth]
    ma = 100 * mc / mt if mt > 0 else 0
    print(f"  {meth:<25s} | {mc:>8d} | {mt:>8d} | {ma:>6.1f}%")

print(f"\n  Top 20 features by volume:")
feat_list_n = [(fn, s['correct'], s['total'], 100*s['correct']/s['total'] if s['total'] > 0 else 0)
               for fn, s in rn['feature_stats'].items()]
feat_list_n.sort(key=lambda x: -x[2])
print(f"  {'Feature':<35s} | {'Corr':>7s} | {'Total':>7s} | {'Acc':>7s}")
print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
for fname, fc, ft, fa in feat_list_n[:20]:
    print(f"  {fname:<35s} | {fc:>7d} | {ft:>7d} | {fa:>6.1f}%")

print(f"\n  Sample errors (first 30):")
for err in rn['errors'][:30]:
    print(f"    [{err['method']:<20s}] {err['feature']:<25s}: pred='{err['predicted'][:35]}' actual='{err['actual'][:35]}'")


print("\n" + "=" * 80)
print("RESULTS — CATEGORICAL FEATURES")
print("=" * 80)
rc = results_cat
print(f"\n  Accuracy: {rc['correct']}/{rc['total']} = {rc['accuracy']:.2f}%")
print(f"  Time:     {rc['elapsed']:.1f}s ({rc['total']/max(rc['elapsed'],0.1):.0f} rows/sec)")
print(f"\n  Method breakdown:")
print(f"  {'Method':<30s} | {'Correct':>8s} | {'Total':>8s} | {'Acc':>7s}")
print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
for meth in sorted(rc['method_stats'].keys(), key=lambda m: -rc['method_stats'][m]):
    mc = rc['method_correct'].get(meth, 0)
    mt = rc['method_stats'][meth]
    ma = 100 * mc / mt if mt > 0 else 0
    print(f"  {meth:<30s} | {mc:>8d} | {mt:>8d} | {ma:>6.1f}%")

print(f"\n  Top 20 features by volume:")
feat_list_c = [(fn, s['correct'], s['total'], 100*s['correct']/s['total'] if s['total'] > 0 else 0)
               for fn, s in rc['feature_stats'].items()]
feat_list_c.sort(key=lambda x: -x[2])
print(f"  {'Feature':<35s} | {'Corr':>7s} | {'Total':>7s} | {'Acc':>7s}")
print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
for fname, fc, ft, fa in feat_list_c[:20]:
    print(f"  {fname:<35s} | {fc:>7d} | {ft:>7d} | {fa:>6.1f}%")

print(f"\n  Sample errors (first 30):")
for err in rc['errors'][:30]:
    print(f"    [{err['method']:<25s}] {err['feature']:<25s}: pred='{err['predicted'][:35]}' actual='{err['actual'][:35]}'")


print("\n" + "=" * 80)
print("COMBINED RESULTS")
print("=" * 80)

print(f"""
{'='*60}
  NUMERIC ACCURACY:      {rn['correct']:>7d}/{rn['total']:<7d} = {rn['accuracy']:>6.2f}%
  CATEGORICAL ACCURACY:  {rc['correct']:>7d}/{rc['total']:<7d} = {rc['accuracy']:>6.2f}%
  ────────────────────────────────────────────────────
  FULL ACCURACY:         {combined_correct:>7d}/{combined_total:<7d} = {combined_accuracy:>6.2f}%
{'='*60}
  Parallel eval time: {eval_elapsed:.1f}s
  Total wall time:    {time.time() - t_total:.1f}s
{'='*60}
""")
