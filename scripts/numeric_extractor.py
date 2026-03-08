"""
Numeric Feature Extractor v4 - Optimized speed + targeted fixes for problem features.
Key improvements:
  - Fast boundary-checked substring matching (no per-value regex compilation)
  - Dimension string position mapping (AxBxC → Ø unten=A, Ø oben=B, Höhe=C)
  - Verpackungseinheit "St." / "Stk" → "Stück" unit alias matching
  - Luftdurchsatz "ab N m³/h" — proximity matching to nearest allowed value
  - Einzelzeichen — letter extraction from title quotes
  - Fächeranzahl — NxM multiplication
  - Spannbereich von/bis — range parsing
"""
import pandas as pd
import numpy as np
import re
import time
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

products_train = pd.read_parquet("data/train/products.parquet")
features_train = pd.read_parquet("data/train/product_features.parquet")
products_val = pd.read_parquet("data/val/products.parquet")
features_val = pd.read_parquet("data/val/product_features.parquet")
taxonomy = pd.read_parquet("data/taxonomy/taxonomy.parquet")

print(f"Train: {len(products_train)} products, {len(features_train)} features")
print(f"Val:   {len(products_val)} products, {len(features_val)} features")

# Create shuffled training set
print("Creating shuffled training set...")
shuffled_products = products_train.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_features = features_train.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_products.to_parquet("data/train/products.shuffled.parquet")
shuffled_features.to_parquet("data/train/product_features.shuffled.parquet")
print("Saved shuffled training data.")

# Filter numeric
num_features_train = features_train[features_train['feature_type'] == 'numeric']
num_features_val = features_val[features_val['feature_type'] == 'numeric']
num_taxonomy = taxonomy[taxonomy['feature_type'] == 'numeric']
print(f"Numeric - Train: {len(num_features_train)}, Val: {len(num_features_val)}, Tax: {len(num_taxonomy)}")

# =============================================================================
# PARSE TAXONOMY
# =============================================================================
print("\nParsing taxonomy...")

def parse_aggregated_values(agg_str):
    if not isinstance(agg_str, str) or not agg_str.strip():
        return []
    return [v.strip() for v in re.findall(r'\[([^\]]*)\]', agg_str) if v.strip()]

tax_values = {}
for _, row in num_taxonomy.iterrows():
    key = (row['category'], row['feature_name'])
    tax_values[key] = parse_aggregated_values(row['aggregated_feature_values'])

print(f"Taxonomy pairs: {len(tax_values)}")

# =============================================================================
# TRAINING PRIORS
# =============================================================================
print("Building training priors...")

train_merged = num_features_train.merge(
    products_train[['uid', 'category']], on='uid', how='left'
)

feat_priors = {}
for fname, grp in train_merged.groupby('feature_name'):
    feat_priors[fname] = grp['feature_value'].value_counts().to_dict()

feat_most_common = {fn: max(vc, key=vc.get) for fn, vc in feat_priors.items()}
print(f"Feature priors: {len(feat_priors)}")


# =============================================================================
# FAST BOUNDARY-CHECKED SUBSTRING MATCH
# =============================================================================
# Characters that cannot precede a number (to prevent "5 l" matching in "55 l")
_BEFORE_NUM_BOUNDARY = set('0123456789')
# Characters that cannot follow the last char if it's a digit
_AFTER_NUM_BOUNDARY = set('0123456789')


def _find_bounded(haystack, needle, needle_starts_with_digit=False, needle_ends_with_digit=False):
    """Check if needle appears in haystack with proper boundaries.
    Returns index of first match or -1."""
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return -1
        # Check leading boundary
        ok = True
        if needle_starts_with_digit and idx > 0:
            ch = haystack[idx - 1]
            if ch in _BEFORE_NUM_BOUNDARY or ch == '.' or ch == ',':
                ok = False
        # Check trailing boundary
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


# =============================================================================
# VALUE MATCHER
# =============================================================================
class ValueMatcher:
    """Pre-compiled matcher for a set of allowed values."""
    
    def __init__(self, allowed_values, feature_name, priors):
        self.allowed = allowed_values
        self.feature_name = feature_name
        self.priors = priors or {}
        
        # Sort longest first for greedy matching
        self.allowed_sorted = sorted(allowed_values, key=lambda v: -len(v))
        self.allowed_lower_sorted = [
            (v, v.lower(), _starts_digit(v), _ends_digit(v))
            for v in self.allowed_sorted
        ]
        
        # Pre-compute tight forms
        self.allowed_tight_sorted = sorted(
            [(v, v.lower().replace(' ', ''), _starts_digit(v), _ends_digit(v))
             for v in allowed_values],
            key=lambda x: -len(x[1])
        )
        
        # Pre-compute num+unit decompositions
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
        
        # Group by unit
        self.by_unit = defaultdict(list)
        for v, num_str, unit, num_f in self.num_unit_pairs:
            self.by_unit[unit.lower()].append((v, num_str, num_f))
        
        # Compile unit regexes (one per unique unit, not per value)
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
        
        # Compile prefix patterns
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
        
        # Pre-compile special patterns
        self.special_patterns = []
        self._compile_specials()
    
    def _compile_specials(self):
        for v in self.allowed:
            # ":N" (Maßstab)
            if re.match(r'^:\d+$', v):
                num = v[1:]
                try:
                    pat = re.compile(r'(?:1\s*:\s*' + re.escape(num) + r'|:\s*' + re.escape(num) + r')(?:\b|\s|[,;)/]|$)')
                    self.special_patterns.append((pat, v))
                except:
                    pass
                continue
            
            # "N°" (Winkel)
            m = re.match(r'^(\d+)°$', v)
            if m:
                try:
                    pat = re.compile(r'(?<!\d)' + re.escape(m.group(1)) + r'\s*°')
                    self.special_patterns.append((pat, v))
                except:
                    pass
                continue
            
            # "N-suffix" (N-polig, N-teilig, N-schneidig)
            m = re.match(r'^(\d+)-(\w+)$', v)
            if m:
                try:
                    pat = re.compile(r'(?<!\d)' + re.escape(m.group(1)) + r'\s*-?\s*' + re.escape(m.group(2).lower()), re.IGNORECASE)
                    self.special_patterns.append((pat, v))
                except:
                    pass
                continue
            
            # "ca. N Unit"
            m = re.match(r'^ca\.\s*([\d\.,]+)\s+(.+)$', v)
            if m:
                try:
                    num_esc = re.escape(m.group(1))
                    unit_esc = re.escape(m.group(2).lower())
                    pat = re.compile(r'(?:ca\.?\s*)?' + num_esc + r'\s*' + unit_esc, re.IGNORECASE)
                    self.special_patterns.append((pat, v))
                except:
                    pass
                continue
            
            # "ab N Unit"
            m = re.match(r'^ab\s+([\d\.,]+)\s+(.+)$', v)
            if m:
                try:
                    pat = re.compile(r'(?:ab\s+)?' + re.escape(m.group(1)) + r'\s*' + re.escape(m.group(2).lower()), re.IGNORECASE)
                    self.special_patterns.append((pat, v))
                except:
                    pass
                continue
            
            # "bis N Unit"
            m = re.match(r'^bis\s+([\d\.,]+)\s+(.+)$', v)
            if m:
                try:
                    pat = re.compile(r'(?:bis\s+)?' + re.escape(m.group(1)) + r'\s*' + re.escape(m.group(2).lower()), re.IGNORECASE)
                    self.special_patterns.append((pat, v))
                except:
                    pass
                continue
            
            # "N x N Unit" (Format)
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
                continue
            
            # "N x N x N Unit"
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

    def _pick_best(self, candidates):
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda v: self.priors.get(v, 0))
    
    def match(self, text_lower, title_lower):
        """Try all strategies. Returns (value, method) or (None, None)."""
        
        # ---- STRATEGY 1: Bounded substring ----
        title_matches = []
        desc_matches = []
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
        
        # ---- STRATEGY 2: Tight match (no spaces) ----
        text_tight = text_lower.replace(' ', '')
        title_tight = title_lower.replace(' ', '')
        for val, val_tight, sd, ed in self.allowed_tight_sorted:
            if len(val_tight) < 3:
                continue
            for src in [title_tight, text_tight]:
                idx = _find_bounded(src, val_tight, sd, ed)
                if idx >= 0:
                    return val, 'tight'
        
        # ---- STRATEGY 3: Special patterns ----
        sp_matches = []
        for pat, val in self.special_patterns:
            if pat.search(text_lower):
                sp_matches.append(val)
        if sp_matches:
            return self._pick_best(list(dict.fromkeys(sp_matches))), 'special'
        
        # ---- STRATEGY 4: Prefix match ----
        prefix_matches = []
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
                        prefix_matches.append(val)
                    elif expected_f is not None and abs(found_f - expected_f) < 0.01:
                        prefix_matches.append(val)
        if prefix_matches:
            return self._pick_best(list(dict.fromkeys(prefix_matches))), 'prefix'
        
        # ---- STRATEGY 5: Number + unit regex ----
        nu_title = []
        nu_desc = []
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
        
        # ---- STRATEGY 6: Dimension string parse ----
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


# =============================================================================
# SPECIAL FEATURE HANDLERS
# =============================================================================

# Regex for dimension strings
_dim3_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)')
_dim2_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)')
_range_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m|µm|kg|g|l|ml)')
_hxbxt_re = re.compile(r'hxbxt\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
_bxhxt_re = re.compile(r'bxhxt\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
_bxtxh_re = re.compile(r'bxtxh\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
_lxbxh_re = re.compile(r'lxbxh\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
_lxb_re = re.compile(r'lxb\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
_bxl_re = re.compile(r'bxl\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', re.IGNORECASE)
# Stück aliases
_stueck_re = re.compile(r'(?<!\d)(\d+)\s*(?:st\.?|stk\.?|stück|pcs)(?:\b|\s|$|[,;)/])', re.IGNORECASE)
_ve_re = re.compile(r'VE\s*[=:]\s*(\d+)\s*(?:st\.?|stk\.?|stück|pcs)?', re.IGNORECASE)
# Fächer multiplication
_faecher_mult_re = re.compile(r'(\d+)\s*[xX×]\s*(\d+)\s*fächer', re.IGNORECASE)
# Letter extraction for Einzelzeichen
_letter_re = re.compile(r'(?:buchstabe|zeichen|letter)\s*["\']?\s*([a-zäöüß])\s*["\']?', re.IGNORECASE)
_letter_re2 = re.compile(r'["\']([a-zäöüß])["\']', re.IGNORECASE)
# Range patterns for Spannbereich
_spann_range_re = re.compile(r'(?<!\d)(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)')
# Volumenstrom for Luftdurchsatz
_volstrom_re = re.compile(r'(\d+(?:[.,]?\d+)?)\s*(?:m³/h|cbm/h)', re.IGNORECASE)
# Sitzhöhe
_sitzhoehe_re = re.compile(r'(?:sitz\w*höhe|sitzhöhe)\s*[:\s]*(\d+(?:[.,]\d+)?)\s*(mm|cm)?', re.IGNORECASE)


def handle_verpackungseinheit(text_lower, title_lower, allowed, priors):
    """Match Verpackungseinheit: 'N Stück' from text with 'St.', 'Stk', etc."""
    allowed_nums = {}
    for v in allowed:
        m = re.match(r'^(\d+)\s+Stück$', v)
        if m:
            allowed_nums[int(m.group(1))] = v
    
    if not allowed_nums:
        return None, None
    
    # Try VE= pattern first
    for m in _ve_re.finditer(text_lower):
        n = int(m.group(1))
        if n in allowed_nums:
            return allowed_nums[n], 'vpe_ve'
    
    # Try "N St." / "N Stk." / "N Stück"
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
    """Extract single letter from title for Einzelzeichen feature."""
    allowed_letters = {}
    for v in allowed:
        m = re.match(r'^([A-Za-zÄÖÜäöüß]+)\s*•?$', v.strip())
        if m:
            allowed_letters[m.group(1).upper()] = v
    
    if not allowed_letters:
        return None, None
    
    # Try pattern: Buchstabe "X" or 'X'
    for pat in [_letter_re, _letter_re2]:
        m = pat.search(title_lower)
        if m:
            letter = m.group(1).upper()
            if letter in allowed_letters:
                return allowed_letters[letter], 'einzelzeichen'
    
    # Try: look for single uppercase letter bounded by punctuation or category marker
    # e.g., ".003.Z/K" → Z, ".003.O" → O
    for m in re.finditer(r'\.(\w)/|\.\d+\.([A-Z])(?:\b|/)', text_lower, re.IGNORECASE):
        letter = (m.group(1) or m.group(2) or '').upper()
        if letter in allowed_letters:
            return allowed_letters[letter], 'einzelzeichen'
    
    # Try code pattern: "NNN.NNN.X" where X is the letter
    for m in re.finditer(r'\d{4,5}\.\d{3}\.([a-z])', title_lower, re.IGNORECASE):
        letter = m.group(1).upper()
        if letter in allowed_letters:
            return allowed_letters[letter], 'einzelzeichen'
    
    return None, None


def handle_faecher(text_lower, title_lower, allowed, priors):
    """Handle Fächeranzahl: NxM Fächer → N*M."""
    allowed_nums = {}
    for v in allowed:
        m = re.match(r'^(\d+)\s+Fächer$', v)
        if m:
            allowed_nums[int(m.group(1))] = v
    
    if not allowed_nums:
        return None, None
    
    # Try "NxM Fächer" multiplication
    for m in _faecher_mult_re.finditer(text_lower):
        product = int(m.group(1)) * int(m.group(2))
        if product in allowed_nums:
            return allowed_nums[product], 'faecher_mult'
    
    # Try direct number + Fächer
    for m in re.finditer(r'(\d+)\s*fächer', text_lower, re.IGNORECASE):
        n = int(m.group(1))
        if n in allowed_nums:
            return allowed_nums[n], 'faecher_direct'
    
    return None, None


def handle_luftdurchsatz(text_lower, title_lower, allowed, priors):
    """Match Luftdurchsatz: find m³/h value, map to nearest 'ab N m³/h'."""
    # Parse allowed values
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
    
    # Find m³/h values in text
    for m in _volstrom_re.finditer(text_lower):
        try:
            found = float(m.group(1).replace(',', '.'))
        except:
            continue
        # Find the closest "ab N m³/h" where found >= N
        best = None
        best_diff = float('inf')
        for val, num in ab_vals:
            diff = abs(found - num)
            if diff < best_diff:
                best_diff = diff
                best = val
        if best and best_diff <= found * 0.15:  # Within 15% tolerance
            return best, 'luft_volstrom'
    
    return None, None


def handle_spannbereich(text_lower, title_lower, allowed, feat_name):
    """Handle Spannbereich von/bis: extract from range patterns like '20-26mm'."""
    is_von = 'von' in feat_name.lower() or 'min' in feat_name.lower()
    
    # Parse allowed values
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
    
    # Find range patterns: N-N unit
    for m in _spann_range_re.finditer(text_lower):
        try:
            n1 = float(m.group(1).replace(',', '.'))
            n2 = float(m.group(2).replace(',', '.'))
        except:
            continue
        target = n1 if is_von else n2
        if target in allowed_nums:
            return allowed_nums[target], 'spann_range'
        # Also try rounded
        for allowed_n, v in allowed_nums.items():
            if abs(target - allowed_n) < 0.1:
                return v, 'spann_range'
    
    return None, None


def handle_dimension_by_name(text_lower, title_lower, feature_name, allowed, priors):
    """
    For dimension-named features, extract from labeled dimension strings.
    HxBxT ==> Höhe x Breite x Tiefe
    BxHxT ==> Breite x Höhe x Tiefe
    LxBxH ==> Länge x Breite x Höhe
    """
    # Map feature names to dimension positions
    fname_lower = feature_name.lower()
    
    # Parse allowed values to get numbers
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
    
    # Try HxBxT pattern
    m = _hxbxt_re.search(text_lower)
    if m:
        dims = {'höhe': m.group(1), 'gesamt_höhe': m.group(1), 'gesamthöhe': m.group(1),
                'breite': m.group(2), 'gesamt_breite': m.group(2), 'gesamtbreite': m.group(2),
                'tiefe': m.group(3), 'gesamt_tiefe': m.group(3), 'gesamttiefe': m.group(3)}
        for key, val_str in dims.items():
            if key in fname_lower.replace('-', '_').replace(' ', '_'):
                try:
                    num = float(val_str.replace(',', '.'))
                    if num in allowed_nums:
                        return allowed_nums[num], 'dim_hxbxt'
                except:
                    pass
    
    # Try BxHxT
    m = _bxhxt_re.search(text_lower)
    if m:
        dims = {'breite': m.group(1), 'gesamtbreite': m.group(1),
                'höhe': m.group(2), 'gesamthöhe': m.group(2),
                'tiefe': m.group(3), 'gesamttiefe': m.group(3)}
        for key, val_str in dims.items():
            if key in fname_lower.replace('-', '_').replace(' ', '_'):
                try:
                    num = float(val_str.replace(',', '.'))
                    if num in allowed_nums:
                        return allowed_nums[num], 'dim_bxhxt'
                except:
                    pass
    
    # Try BxTxH
    m = _bxtxh_re.search(text_lower)
    if m:
        dims = {'breite': m.group(1), 'gesamtbreite': m.group(1),
                'tiefe': m.group(2), 'gesamttiefe': m.group(2),
                'höhe': m.group(3), 'gesamthöhe': m.group(3)}
        for key, val_str in dims.items():
            if key in fname_lower.replace('-', '_').replace(' ', '_'):
                try:
                    num = float(val_str.replace(',', '.'))
                    if num in allowed_nums:
                        return allowed_nums[num], 'dim_bxtxh'
                except:
                    pass
    
    # Try LxBxH
    m = _lxbxh_re.search(text_lower)
    if m:
        dims = {'länge': m.group(1), 'gesamtlänge': m.group(1),
                'breite': m.group(2), 'gesamtbreite': m.group(2),
                'höhe': m.group(3), 'gesamthöhe': m.group(3)}
        for key, val_str in dims.items():
            if key in fname_lower.replace('-', '_').replace(' ', '_'):
                try:
                    num = float(val_str.replace(',', '.'))
                    if num in allowed_nums:
                        return allowed_nums[num], 'dim_lxbxh'
                except:
                    pass
    
    # Try LxB
    m = _lxb_re.search(text_lower)
    if m:
        dims = {'länge': m.group(1), 'gesamtlänge': m.group(1),
                'breite': m.group(2), 'gesamtbreite': m.group(2)}
        for key, val_str in dims.items():
            if key in fname_lower.replace('-', '_').replace(' ', '_'):
                try:
                    num = float(val_str.replace(',', '.'))
                    if num in allowed_nums:
                        return allowed_nums[num], 'dim_lxb'
                except:
                    pass
    
    # Try BxL
    m = _bxl_re.search(text_lower)
    if m:
        dims = {'breite': m.group(1), 'gesamtbreite': m.group(1),
                'länge': m.group(2), 'gesamtlänge': m.group(2)}
        for key, val_str in dims.items():
            if key in fname_lower.replace('-', '_').replace(' ', '_'):
                try:
                    num = float(val_str.replace(',', '.'))
                    if num in allowed_nums:
                        return allowed_nums[num], 'dim_bxl'
                except:
                    pass
    
    return None, None


def handle_sitzhöhe(text_lower, title_lower, allowed, priors):
    """Extract Sitzhöhe from description text."""
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
                n *= 10  # Convert to mm
            if n in allowed_nums:
                return allowed_nums[n], 'sitzhoehe'
        except:
            pass
    
    return None, None


# =============================================================================
# PRE-COMPILE ALL MATCHERS
# =============================================================================
print("Pre-compiling matchers...")
t_compile = time.time()
matchers = {}
for (cat, fname), vals in tax_values.items():
    if vals:
        priors = feat_priors.get(fname, {})
        matchers[(cat, fname)] = ValueMatcher(vals, fname, priors)
print(f"Compiled {len(matchers)} matchers in {time.time()-t_compile:.1f}s")


# =============================================================================
# TEXT CLEANING
# =============================================================================
_tag_re = re.compile(r'<[^>]+>')
_space_re = re.compile(r'\s+')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('<br>', ' ').replace('<br/>', ' ').replace('<br />', ' ')
    text = text.replace('<BR>', ' ').replace('<BR/>', ' ')
    text = _tag_re.sub(' ', text)
    text = _space_re.sub(' ', text)
    return text.strip()


# =============================================================================
# FEATURE NAME SETS FOR SPECIAL HANDLERS
# =============================================================================
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

# Features that should try special handlers FIRST (before generic matcher)
SPECIAL_FIRST_FEATURES = {
    'Verpackungseinheit', 'Einzelzeichen', 'Fächeranzahl',
    'Luftdurchsatz', 'Sitzhöhe',
    'Ø unten', 'Ø oben', 'Höhe',
    'Innen-Ø', 'Außen-Ø', 'Kopf-Ø',
    'Messbereich von', 'Messbereich bis',
    'Min. Volumen', 'Max. Volumen',
} | SPANN_FEATURES | DIM_FEATURES


# =============================================================================
# DIMENSION POSITION HANDLER (for gummi_vollstopfen and similar)
# =============================================================================
# Map category patterns to dimension feature position mappings
# In "AxBxC unit": position 0=A, 1=B, 2=C

# Build position maps from training data
print("Building dimension position maps from training data...")
_dim_pos_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)')
_dim_pos_re2 = re.compile(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)')

def _parse_dim_num(s):
    try:
        return float(s.replace(',', '.'))
    except:
        return None

def handle_dim_position(text_lower, title_lower, feature_name, allowed, category):
    """
    For features like Ø unten, Ø oben, Höhe in gummi_vollstopfen:
    Parse dimension strings AxBxC and map by position.
    Also handles Innen-Ø/Außen-Ø for washers (NxNxN → Innen, Außen, Dicke)
    and Kopf-Ø for tools (first dimension in NxNxN).
    """
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
    
    # Determine position index for this feature
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
    
    # Try 3D dimensions first in text
    for m in _dim_pos_re.finditer(text_lower):
        dims = [m.group(1), m.group(2), m.group(3)]
        if target_pos < len(dims):
            n = _parse_dim_num(dims[target_pos])
            if n in allowed_nums:
                return allowed_nums[n], 'dim_position'
    
    # Try title specifically
    for m in _dim_pos_re.finditer(title_lower):
        dims = [m.group(1), m.group(2), m.group(3)]
        if target_pos < len(dims):
            n = _parse_dim_num(dims[target_pos])
            if n in allowed_nums:
                return allowed_nums[n], 'dim_position'
    
    # For Innen-Ø/Außen-Ø, also try "N mm N mm" pattern (washer titles)
    if fname_l in ('innen-ø', 'außen-ø'):
        # Pattern: "Unterlegscheibe 5.3 mm 10 mm" → first=Innen, second=Außen
        washer_pat = re.compile(r'(?:scheibe|washer)\w*\s+([\d\.,]+)\s*mm\s+([\d\.,]+)\s*mm', re.IGNORECASE)
        for m in washer_pat.finditer(text_lower):
            pos = 0 if fname_l.startswith('innen') else 1
            dims_str = [m.group(1), m.group(2)]
            n = _parse_dim_num(dims_str[pos])
            if n in allowed_nums:
                return allowed_nums[n], 'dim_washer'
        
        # Also try "ISO7089 NxNxN" from title  
        iso_pat = re.compile(r'iso\d+\s+([\d\.,]+)\s*[xX×]\s*([\d\.,]+)\s*[xX×]\s*([\d\.,]+)', re.IGNORECASE)
        for m in iso_pat.finditer(text_lower):
            pos = 0 if fname_l.startswith('innen') else 1
            n = _parse_dim_num(m.group(pos + 1))
            if n in allowed_nums:
                return allowed_nums[n], 'dim_washer'
    
    # For Kopf-Ø, try 2D dimensions too
    if fname_l == 'kopf-ø' and target_pos == 0:
        for m in _dim_pos_re2.finditer(title_lower):
            n = _parse_dim_num(m.group(1))
            if n in allowed_nums:
                return allowed_nums[n], 'dim_position'
    
    return None, None


def handle_range_feature(text_lower, title_lower, allowed, feat_name):
    """Handle features that extract values from ranges like '20-26mm', 'M8-M20', etc."""
    is_min = any(k in feat_name.lower() for k in ['von', 'min', 'von '])
    is_max = any(k in feat_name.lower() for k in ['bis', 'max', 'bis '])
    
    # Parse allowed values - store as list per numeric key to handle duplicates
    allowed_nums = defaultdict(list)
    for v in allowed:
        m = re.match(r'^([\d\.,]+)\s+(.+)$', v)
        if m:
            n = _parse_dim_num(m.group(1))
            if n is not None:
                allowed_nums[n].append(v)
    
    if not allowed_nums:
        return None, None
    
    def _pick_best_format(vals, raw_str):
        """From multiple format variants, pick the one matching the raw text number best."""
        if len(vals) == 1:
            return vals[0]
        # Prefer integer format "25 mm" over "25.0 mm" when text number is integer
        if '.' not in raw_str:
            for v in vals:
                m = re.match(r'^(\d+)\s', v)
                if m:
                    return v
        else:
            for v in vals:
                if '.' in v.split()[0] or ',' in v.split()[0]:
                    return v
        return vals[0]
    
    # Try multiple range patterns
    range_patterns = [
        # N-N unit
        re.compile(r'(?<!\d)([\d\.,]+)\s*[-–—]\s*([\d\.,]+)\s*(mm|cm|m|µm|µl|ml|l|kg|g|bar|°C|A|V|W|kW|kN|N|nm|µm|g/ml|g/cm³|g/cm3)', re.IGNORECASE),
        # N bis N unit
        re.compile(r'(?<!\d)([\d\.,]+)\s+bis\s+([\d\.,]+)\s*(mm|cm|m|µm|µl|ml|l|kg|g|bar|°C|A|V|W|kW|kN|N|nm|µm|g/ml|g/cm³|g/cm3)', re.IGNORECASE),
        # N ... N (ellipsis separator)
        re.compile(r'(?<!\d)([\d\.,]+)\s*\.{2,3}\s*([\d\.,]+)', re.IGNORECASE),
        # N-N:step format (like "1,420-1,480:0,001")
        re.compile(r'(?<!\d)([\d\.,]+)\s*[-–—]\s*([\d\.,]+)\s*:\s*[\d\.,]+', re.IGNORECASE),
        # N - N (no unit, just range with spaces)
        re.compile(r'(?<!\d)([\d\.,]+)\s*[-–—]\s*([\d\.,]+)(?=\s|$|[,;)/])', re.IGNORECASE),
    ]
    
    for pat in range_patterns:
        for m in pat.finditer(text_lower):
            n1 = _parse_dim_num(m.group(1))
            n2 = _parse_dim_num(m.group(2))
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
                return _pick_best_format(allowed_nums[target], raw), 'range_parse'
            # Try close match
            for an, avs in allowed_nums.items():
                if abs(target - an) < 0.1:
                    return _pick_best_format(avs, raw), 'range_parse'
    
    return None, None



# =============================================================================
# PREDICTION FUNCTION (importable)
# =============================================================================
def predict_numeric_row(title_lower, full_lower, feature_name, category):
    """Predict a single numeric feature value. Returns string or None."""
    key = (category, feature_name)
    allowed = tax_values.get(key, [])
    priors_for_feat = feat_priors.get(feature_name, {})
    predicted = None
    method = 'no_match'

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
            predicted, method = handle_sitzhöhe(full_lower, title_lower, allowed, priors_for_feat)
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
        matcher = matchers.get(key)
        if matcher:
            predicted, method = matcher.match(full_lower, title_lower)

    # Remaining range features
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

    return str(predicted).strip() if predicted else None


if __name__ == '__main__':
    # =============================================================================
    # EVALUATION
    # =============================================================================
    print("\n" + "=" * 80)
    print("EVALUATING ON VALIDATION SET")
    print("=" * 80)

    val_merged = num_features_val.merge(
        products_val[['uid', 'category', 'title', 'description']],
        on='uid', how='left'
    )
    print(f"Rows to evaluate: {len(val_merged)}")

    # Pre-clean texts (vectorized)
    print("Pre-cleaning texts...")
    t_clean = time.time()
    titles_clean = products_val['title'].fillna('').apply(clean_text).str.lower()
    descs_clean = products_val['description'].fillna('').apply(clean_text).str.lower()
    fulls = titles_clean + " " + descs_clean
    uid_text_cache = dict(zip(products_val['uid'], zip(titles_clean, fulls)))
    print(f"Cached {len(uid_text_cache)} products in {time.time()-t_clean:.1f}s")

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

        predicted = None
        method = 'no_match'

        text_data = uid_text_cache.get(uid)
        if text_data is None:
            title_lower = clean_text(row.get('title', '')).lower()
            full_lower = title_lower + " " + clean_text(row.get('description', '')).lower()
        else:
            title_lower, full_lower = text_data

        key = (category, feature_name)
        allowed = tax_values.get(key, [])
        priors_for_feat = feat_priors.get(feature_name, {})

        # --- TRY SPECIAL HANDLERS FIRST for targeted features ---
        if allowed and feature_name in SPECIAL_FIRST_FEATURES:
            # Verpackungseinheit
            if feature_name == 'Verpackungseinheit':
                predicted, method = handle_verpackungseinheit(full_lower, title_lower, allowed, priors_for_feat)

            # Einzelzeichen
            elif feature_name == 'Einzelzeichen':
                predicted, method = handle_einzelzeichen(full_lower, title_lower, allowed, priors_for_feat)

            # Fächeranzahl
            elif feature_name == 'Fächeranzahl':
                predicted, method = handle_faecher(full_lower, title_lower, allowed, priors_for_feat)

            # Luftdurchsatz
            elif feature_name == 'Luftdurchsatz':
                predicted, method = handle_luftdurchsatz(full_lower, title_lower, allowed, priors_for_feat)

            # Sitzhöhe
            elif feature_name == 'Sitzhöhe':
                predicted, method = handle_sitzhöhe(full_lower, title_lower, allowed, priors_for_feat)

            # Ø unten / Ø oben / Höhe (dimension position mapping for gummi_vollstopfen etc.)
            elif feature_name in ('Ø unten', 'Ø oben', 'Höhe'):
                predicted, method = handle_dim_position(full_lower, title_lower, feature_name, allowed, category)
                # Also try labeled dimension strings (HxBxT etc.)
                if predicted is None and feature_name in DIM_FEATURES:
                    predicted, method = handle_dimension_by_name(full_lower, title_lower, feature_name, allowed, priors_for_feat)

            # Innen-Ø / Außen-Ø / Kopf-Ø (dimension position for washers, tools)
            elif feature_name in ('Innen-Ø', 'Außen-Ø', 'Kopf-Ø'):
                predicted, method = handle_dim_position(full_lower, title_lower, feature_name, allowed, category)

            # Spannbereich / range features (including Messbereich, Min/Max Volumen)
            elif feature_name in SPANN_FEATURES or feature_name in ('Messbereich von', 'Messbereich bis', 'Min. Volumen', 'Max. Volumen'):
                predicted, method = handle_range_feature(full_lower, title_lower, allowed, feature_name)

            # Dimension-based features (HxBxT, etc.)
            elif feature_name in DIM_FEATURES:
                predicted, method = handle_dimension_by_name(full_lower, title_lower, feature_name, allowed, priors_for_feat)
                # If dimension_by_name didn't find, try range parsing (for Höhe etc that appear in ranges)
                if predicted is None:
                    predicted, method = handle_range_feature(full_lower, title_lower, allowed, feature_name)

        # --- TRY MAIN MATCHER (if special handler didn't find anything) ---
        if predicted is None:
            matcher = matchers.get(key)
            if matcher:
                predicted, method = matcher.match(full_lower, title_lower)

        # --- TRY REMAINING SPECIAL HANDLERS (for features not in SPECIAL_FIRST) ---  
        if predicted is None and allowed and feature_name not in SPECIAL_FIRST_FEATURES:
            # Any remaining range-like features we didn't catch
            fname_l = feature_name.lower()
            if any(k in fname_l for k in ['von', 'bis', 'min', 'max', 'bereich']):
                predicted, method = handle_range_feature(full_lower, title_lower, allowed, feature_name)

        # --- FALLBACKS ---
        if predicted is None:
            # Single allowed value
            if len(allowed) == 1:
                predicted = allowed[0]
                method = 'single_value'

        if predicted is None and allowed:
            # Category-derived
            cat_clean = category.lower().replace('_', '').replace('-', '')
            for val in allowed:
                if val.lower().replace(' ', '') in cat_clean:
                    predicted = val
                    method = 'cat_derived'
                    break

        if predicted is None:
            # Training prior (feature_name level, intersected with allowed)
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
            if len(errors) < 500:
                errors.append({
                    'feature': feature_name,
                    'actual': actual,
                    'predicted': pred_str,
                    'method': method or 'no_match',
                    'category': category,
                    'title': title_lower[:120],
                })

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            acc = 100 * correct / total
            print(f"  [{i+1:>7d}/{len(val_merged)}] acc={acc:.1f}% rate={rate:.0f}/s elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    accuracy = 100 * correct / total if total > 0 else 0

    # =============================================================================
    # RESULTS
    # =============================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\n{'='*60}")
    print(f"  OVERALL NUMERIC ACCURACY: {correct}/{total} = {accuracy:.2f}%")
    print(f"  Time: {elapsed:.1f}s ({total/elapsed:.0f} products/sec)")
    print(f"{'='*60}")

    print(f"\nAccuracy by method:")
    print(f"  {'Method':<25s} | {'Correct':>8s} | {'Total':>8s} | {'Acc':>7s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
    for meth in sorted(method_stats.keys(), key=lambda m: -method_stats[m]):
        mc = method_correct.get(meth, 0)
        mt = method_stats[meth]
        ma = 100 * mc / mt if mt > 0 else 0
        print(f"  {meth:<25s} | {mc:>8d} | {mt:>8d} | {ma:>6.1f}%")

    print(f"\nTop 30 features by volume:")
    feat_list = [(fn, s['correct'], s['total'], 100*s['correct']/s['total'] if s['total'] > 0 else 0)
                 for fn, s in feature_stats.items()]
    feat_list.sort(key=lambda x: -x[2])
    print(f"  {'Feature':<35s} | {'Corr':>7s} | {'Total':>7s} | {'Acc':>7s}")
    print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for fname, fc, ft, fa in feat_list[:30]:
        print(f"  {fname:<35s} | {fc:>7d} | {ft:>7d} | {fa:>6.1f}%")

    print(f"\nWorst 20 features (>100 samples):")
    feat_low = [(f, c, t, a) for f, c, t, a in feat_list if t > 100]
    feat_low.sort(key=lambda x: x[3])
    for fname, fc, ft, fa in feat_low[:20]:
        print(f"  {fname:<35s} | {fc:>7d} | {ft:>7d} | {fa:>6.1f}%")

    print(f"\nSample errors (first 50):")
    for err in errors[:50]:
        m = err.get('method') or 'no_match'
        print(f"  [{m:<20s}] {err['feature']:<25s}: pred='{err['predicted'][:40]:<40s}' actual='{err['actual'][:40]}' | cat={err['category'][:30]}")

    # Feature accuracy distribution
    feat_accs = [a for _, _, t, a in feat_list if t > 10]
    print(f"\nFeature accuracy distribution (features with >10 samples):")
    for threshold in [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]:
        cnt = sum(1 for a in feat_accs if a >= threshold)
        print(f"  >={threshold}%: {cnt}/{len(feat_accs)} features")

    print(f"\n{'='*60}")
    print(f"TARGET: >60%")
    print(f"RESULT: {accuracy:.2f}% ({'PASSED' if accuracy >= 60 else 'NOT MET'})")
    print(f"{'='*60}")
