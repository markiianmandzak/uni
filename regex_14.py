import argparse
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

import pandas as pd
from tqdm import tqdm


DATA_ROOT = "data/feature-normalization-hackathon/data"

UNIT_PATTERN = r"m³/h|l/min|g/cm³|g/ml|mm²|m²|°C|µm|mm|cm|kg|mg|ml|µl|cl|bar|Hz|V|W|A|m|g|l|L|°|%|\""

UNIT_FACTORS = {
    "length": {"µm": 0.001, "mm": 1.0, "cm": 10.0, "m": 1000.0},
    "weight": {"mg": 0.001, "g": 1.0, "kg": 1000.0},
    "volume": {"µl": 0.001, "ml": 1.0, "cl": 10.0, "l": 1000.0, "L": 1000.0},
}

UNIT_CANONICAL = {
    "µm": "µm",
    "mm²": "mm²",
    "mm": "mm",
    "cm": "cm",
    "m³/h": "m³/h",
    "m²": "m²",
    "m": "m",
    "g/cm³": "g/ml",
    "g/ml": "g/ml",
    "kg": "kg",
    "g": "g",
    "mg": "mg",
    "ml": "ml",
    "µl": "µl",
    "cl": "cl",
    "l/min": "l/min",
    "l": "l",
    "bar": "bar",
    "v": "V",
    "w": "W",
    "a": "A",
    "hz": "Hz",
    "°c": "°C",
    "°": "°",
    "%": "%",
    '"': '"',
}

DIMENSION_FEATURES = {
    "Länge": "L",
    "Gesamtlänge": "L",
    "Breite": "B",
    "Gesamtbreite": "B",
    "Höhe": "H",
    "Gesamthöhe": "H",
    "Tiefe": "T",
    "Gesamttiefe": "T",
    "Durchmesser": "D",
    "Außen-Ø": "DA",
    "Innen-Ø": "DI",
    "Kopf-Ø": "DK",
    "Nenn-Ø": "D",
    "Rad-Ø": "D",
    "Ladeflächenlänge": "L",
    "Ladeflächenbreite": "B",
    "Spannutenlänge": "L",
    "Kopfhöhe": "H",
    "Kopflänge": "L",
}

FEATURE_ALIASES = {
    "Länge": ["länge", "gesamtlänge", "l", "lang"],
    "Gesamtlänge": ["gesamtlänge", "länge", "l", "lang"],
    "Breite": ["breite", "b"],
    "Gesamtbreite": ["gesamtbreite", "breite", "b"],
    "Höhe": ["höhe", "h"],
    "Gesamthöhe": ["gesamthöhe", "höhe", "h"],
    "Tiefe": ["tiefe", "t"],
    "Gesamttiefe": ["gesamttiefe", "tiefe", "t"],
    "Durchmesser": ["durchmesser", "ø", "dm"],
    "Außen-Ø": ["außen-ø", "außendurchmesser", "da"],
    "Innen-Ø": ["innen-ø", "innendurchmesser", "di", "id"],
    "Kopf-Ø": ["kopf-ø", "kopfdurchmesser"],
    "Gewinde-Ø": ["gewinde", "gewinde-ø", "gewinde ø", "m"],
    "Nenn-Ø": ["nenn-ø", "nennweite", "ø"],
    "Verpackungseinheit": ["ve", "verpackungseinheit", "gebinde"],
    "Körnung": ["körnung", "koernung", "k"],
    "Luftdurchsatz": ["luftdurchsatz", "luftleistung", "förderleistung"],
    "Format": ["format", "din"],
    "Ladeflächenlänge": ["ladeflächenlänge", "ladeflaechenlaenge", "l"],
    "Ladeflächenbreite": ["ladeflächenbreite", "ladeflaechenbreite", "b"],
    "Spannutenlänge": ["spannutenlänge", "spannutenlaenge", "l"],
    "Kopfhöhe": ["kopfhöhe", "kopfhoehe", "h"],
    "Kopflänge": ["kopflänge", "kopflaenge", "l"],
}

FUZZY_CATEGORICAL_FEATURES = {
    "Material",
    "Farbe",
    "Oberfläche",
    "Ausführung",
    "Schleifstoff",
    "Felgenmaterial",
    "Laufbelag",
    "Form",
    "Antrieb",
    "Format",
}

MEASUREMENT_RE = re.compile(
    rf"(?<![\w/])(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>{UNIT_PATTERN})\b",
    re.IGNORECASE,
)

TUPLE_UNIT_PATTERN = r'µm|mm|cm|m|"'

COMMON_TUPLE_RE = re.compile(
    rf"(?<!\d)(?P<a>\d+(?:[.,]\d+)?)(?:\s*(?P<ua>{TUPLE_UNIT_PATTERN})\s*)?x\s*(?P<b>\d+(?:[.,]\d+)?)(?:\s*(?P<ub>{TUPLE_UNIT_PATTERN})\s*)?(?:x\s*(?P<c>\d+(?:[.,]\d+)?)(?:\s*(?P<uc>{TUPLE_UNIT_PATTERN})\s*)?)?(?:x\s*(?P<d>\d+(?:[.,]\d+)?)(?:\s*(?P<ud>{TUPLE_UNIT_PATTERN})\s*)?)?(?:\s*(?P<trail>{TUPLE_UNIT_PATTERN}))?",
    re.IGNORECASE,
)

GENERIC_DIMENSION_RE = re.compile(
    rf"(?<![\w/])(?P<a>\d+(?:[.,]\d+)?)(?:\s*(?P<ua>{TUPLE_UNIT_PATTERN}))?\s*x\s*(?P<b>\d+(?:[.,]\d+)?)(?:\s*(?P<ub>{TUPLE_UNIT_PATTERN}))?(?:\s*x\s*(?P<c>\d+(?:[.,]\d+)?)(?:\s*(?P<uc>{TUPLE_UNIT_PATTERN}))?)?(?:\s*x\s*(?P<d>\d+(?:[.,]\d+)?)(?:\s*(?P<ud>{TUPLE_UNIT_PATTERN}))?)?(?:\s*(?P<trail>{TUPLE_UNIT_PATTERN}))?(?![\w/])",
    re.IGNORECASE,
)

ORDERED_TUPLE_PATTERNS = [
    (re.compile(r"(?i)\bH\s*x\s*B\s*x\s*T\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,40})"), ["H", "B", "T"]),
    (re.compile(r"(?i)\bB\s*x\s*T\s*x\s*H\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,40})"), ["B", "T", "H"]),
    (re.compile(r"(?i)\bL\s*x\s*B\s*x\s*H\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,40})"), ["L", "B", "H"]),
    (re.compile(r"(?i)\bL\s*x\s*B\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,30})"), ["L", "B"]),
    (re.compile(r"(?i)\bB\s*x\s*H\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,30})"), ["B", "H"]),
    (re.compile(r"(?i)\bB\s*x\s*T\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,30})"), ["B", "T"]),
    (re.compile(r"(?i)\bØ\s*x\s*L\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,30})"), ["D", "L"]),
    (re.compile(r"(?i)\bØ\s*x\s*B\b[^\d]{0,12}(?P<vals>[^,;\n<>()]{3,30})"), ["D", "B"]),
]


COMPACT_DIMENSION_PATTERNS = [
    (re.compile(r'(?i)\bh\s*x\s*b\s*x\s*t\b\s*(\d+(?:[.,]\d+)?)x(\d+(?:[.,]\d+)?)x(\d+(?:[.,]\d+)?)(mm|cm|m|")'), {"Höhe": 0, "Gesamthöhe": 0, "Breite": 1, "Gesamtbreite": 1, "Tiefe": 2, "Gesamttiefe": 2}),
    (re.compile(r'(?i)\bb\s*x\s*t\s*x\s*h\b\s*(\d+(?:[.,]\d+)?)x(\d+(?:[.,]\d+)?)x(\d+(?:[.,]\d+)?)(mm|cm|m|")'), {"Breite": 0, "Gesamtbreite": 0, "Tiefe": 1, "Gesamttiefe": 1, "Höhe": 2, "Gesamthöhe": 2}),
]

SCREW_PAIR_RE = re.compile(r'(?i)(?:\bm\s*)?(\d+(?:[.,]\d+)?)\s*x\s*(\d+(?:[.,]\d+)?)(?:\s*(mm|cm|m|"))?')
DURCHMESSER_RE = re.compile(r'(?i)(?:durchmesser|ø)\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(µm|mm|cm|m|")\b')
PACKAGING_PATTERNS = [
    re.compile(r'(?i)\bve\s*[=:]?\s*(\d{1,5})\b'),
    re.compile(r'(?i)\b(\d{1,5})\s*(st\.?|stück|rolle|karton|pack|paar|set)\b'),
]
THREAD_PATTERNS = [
    re.compile(r'(?i)\b(M\s*\d+(?:[.,]\d+)?)(?=(?:\s*[x/]|\b))'),
    re.compile(r'(?i)\b(ST\s*\d+(?:[.,]\d+)?)(?=(?:\s*[x/]|\b))'),
    re.compile(r'(?i)\b(G\s*\d+(?:[.,]\d+)?/\d+\s*"?)(?=(?:\s*[x/]|\b))'),
    re.compile(r'(?i)\b(\d+(?:[.,]\d+)?/\d+\s*"?)(?=(?:\s*[x/]|\b))'),
]
RANGE_PATTERNS = [
    re.compile(r'(?i)(?:spannbereich|messbereich)\s*(\d+(?:[.,]\d+)?)\s*(?:\.\.\.|-|bis)?\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m|")'),
    re.compile(r'(?i)(\d+(?:[.,]\d+)?)\s*(?:\.\.\.|-)\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m|")'),
]
DENSITY_RANGE_PATTERNS = [
    re.compile(r'(?i)(?:messbereich|dichte[- ]?aräometer|hydrometer|specific gravity)[^\d]{0,20}(\d+(?:[.,]\d+)?)\s*(?:\.\.\.|-|bis|:)\s*(\d+(?:[.,]\d+)?)\s*(g/cm³|g/ml)'),
    re.compile(r'(?i)(\d+(?:[.,]\d+)?)\s*(?:\.\.\.|-|:)\s*(\d+(?:[.,]\d+)?)\s*(g/cm³|g/ml)'),
]
FORMAT_DIMENSION_RE = re.compile(r'(?i)(\d+(?:[.,]\d+)?)\s*x\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)')
FORMAT_PLAIN_RE = re.compile(r'(?<!\d)(\d{2,4})\s*x\s*(\d{2,4})(?!\d)')
BILDDIAGONALE_INCH_RE = re.compile(r'(?i)(\d+(?:[.,]\d+)?)\s*(?:zoll|inch|\")')
BILDDIAGONALE_CM_RE = re.compile(r'(?i)(\d+(?:[.,]\d+)?)\s*cm')
NO_RULE_BETRIEBSDAUER_RE = re.compile(r'(?i)(\d+(?:[.,]\d+)?)\s*min\b')

ALIAS_FEATURES = {"Ausführung", "Material", "Bodenausführung", "Form", "Warenzustand", "Kopfform", "Korpusfarbe", "Frontfarbe", "Farbe"}
ALIAS_TOKEN_RE = re.compile(r"[a-z0-9äöüß]+", re.IGNORECASE)
ALIAS_STOPWORDS = {
    "und", "oder", "der", "die", "das", "des", "den", "dem", "ein", "eine", "einer", "einem", "einen",
    "mit", "ohne", "für", "fuer", "aus", "von", "zum", "zur", "im", "in", "am", "an", "auf", "bei",
    "artikel", "produkt", "set", "stück", "st", "ve", "pack", "karton", "rolle", "paar", "farbe",
    "material", "form", "ausführung", "format", "mm", "cm", "m", "kg", "g", "ml", "l", "ral",
}

SPECIAL_RULES = {
    "Material": [
        (re.compile(r'(?i)stahl\s*verzinkt|galv(?:anisch)?\s*verzinkt'), 'Stahl verzinkt'),
        (re.compile(r'(?i)edelstahl\s*a4|\ba4\b|1\.4401|1\.4571'), 'Edelstahl (A4)'),
        (re.compile(r'(?i)edelstahl\s*a2|\ba2\b|1\.4301'), 'Edelstahl (A2)'),
        (re.compile(r'(?i)edelstahl\s*a1|\ba1\b|1\.4305'), 'Edelstahl (A1)'),
        (re.compile(r'(?i)\ba4\b'), 'A4'),
        (re.compile(r'(?i)\ba2\b'), 'A2'),
        (re.compile(r'(?i)\ba1\b'), 'A1'),
        (re.compile(r'(?i)stahl\s*10[,.]?9|\b10[,.]?9\b'), 'Stahl 10.9'),
        (re.compile(r'(?i)stahl\s*8[,.]?8|\b8[,.]?8\b'), 'Stahl 8.8'),
        (re.compile(r'(?i)stahl\s*5[,.]?8|\b5[,.]?8\b'), 'Stahl 5.8'),
        (re.compile(r'(?i)polyamid'), 'Polyamid'),
        (re.compile(r'(?i)polypropylen'), 'Polypropylen'),
        (re.compile(r'(?i)polycarbonat'), 'Polycarbonat'),
        (re.compile(r'(?i)kunststoff|nylon'), 'Kunststoff'),
        (re.compile(r'(?i)messing|\bms\b'), 'Messing'),
        (re.compile(r'(?i)holz'), 'Holz'),
        (re.compile(r'(?i)edelstahl'), 'Edelstahl'),
        (re.compile(r'(?i)stahl|metall'), 'Metall'),
        (re.compile(r'(?i)stahl'), 'Stahl'),
        (re.compile(r'(?i)aluminium|\balu\b'), 'Aluminium'),
    ],
    "ColorBasic": [
        (re.compile(r'(?i)farblich sortiert|mehrfarbig'), 'mehrfarbiges Set'),
        (re.compile(r'(?i)\bcyan\b'), 'cyan'),
        (re.compile(r'(?i)\borange\b'), 'orange'),
        (re.compile(r'(?i)\bweiß\b|\bweiss\b|reinweiß'), 'weiß'),
        (re.compile(r'(?i)\bschwarz\b'), 'schwarz'),
    ],
    "Oberfläche": [
        (re.compile(r'(?i)galv(?:anisch)?\s*verzinkt'), 'galvanisch verzinkt'),
        (re.compile(r'(?i)gelb\s*verzinkt|verzinkt\s*gelb'), 'verzinkt gelb'),
        (re.compile(r'(?i)\bzink\b|\bverzinkt\b'), 'verzinkt'),
        (re.compile(r'(?i)blank|edelstahl|\ba2\b|\ba4\b'), 'blank'),
    ],
    "Ausführung": [
        (re.compile(r'(?i)verschlu(?:ss|ß)kappe|\bkappe\b|endkappe'), 'Endkappe'),
        (re.compile(r'(?i)deckenwinkel'), 'Deckenwinkel'),
        (re.compile(r'(?i)schiebemuffe'), 'Schiebemuffe'),
        (re.compile(r'(?i)winkelverschraubung'), 'Winkelverschraubung'),
        (re.compile(r'(?i)\bt-?stück\b'), 'T-Stück'),
        (re.compile(r'(?i)selbstklebetasche'), 'Selbstklebetasche'),
        (re.compile(r'(?i)standardhülle'), 'Standardhülle'),
        (re.compile(r'(?i)\bwinkel\b'), 'Winkel'),
        (re.compile(r'(?i)\bmuffe\b'), 'Muffe'),
    ],
    "Anschluss": [
        (re.compile(r'(?i)mini\s*displayport'), 'Mini Displayport'),
        (re.compile(r'(?i)displayport'), 'Displayport'),
        (re.compile(r'(?i)\bdvi\b'), 'DVI'),
        (re.compile(r'(?i)\busb\b'), 'USB'),
        (re.compile(r'(?i)rs\s*422'), 'RS422'),
        (re.compile(r'(?i)rs\s*232'), 'RS232'),
        (re.compile(r'(?i)bluetooth'), 'Bluetooth'),
    ],
    "Felgenmaterial": [
        (re.compile(r'(?i)aluminium'), 'Aluminium'),
        (re.compile(r'(?i)polypropylen|\btpa\b'), 'Polypropylen'),
        (re.compile(r'(?i)polyamid|\bpath\b'), 'Polyamid'),
        (re.compile(r'(?i)stahl'), 'Stahl'),
    ],
    "Laufbelag": [
        (re.compile(r'(?i)thermoplast|\btpa\b'), 'Thermoplast'),
        (re.compile(r'(?i)polyurethan|\bpath\b'), 'Polyurethan'),
        (re.compile(r'(?i)gummi'), 'Gummi'),
        (re.compile(r'(?i)kunststoff'), 'Kunststoff'),
    ],
    "Für Modell": [
        (re.compile(r'(?i)raspberry\s*pi[^a-z0-9]{0,4}2\s*b'), '2B'),
        (re.compile(r'(?i)odroid\s*c2'), 'C2'),
        (re.compile(r'(?i)odroid\s*c1\+'), 'C1+'),
        (re.compile(r'(?i)raspberry\s*pi[^0-9]*5(?![0-9a-z])'), '5'),
        (re.compile(r'(?i)raspberry\s*pi[^0-9]*4(?![0-9a-z])'), '4'),
        (re.compile(r'(?i)raspberry\s*pi[^0-9]*3(?![0-9a-z])'), '3'),
    ],
    "für Modelle von": [
        (re.compile(r'(?i)zebra'), 'Zebra'),
        (re.compile(r'(?i)honeywell'), 'Honeywell'),
        (re.compile(r'(?i)mobile'), 'Mobile'),
    ],
    "Displayoberfläche": [
        (re.compile(r'(?i)glänzend|glossy'), 'glänzend'),
        (re.compile(r'(?i)matt|matte'), 'matt'),
    ],
    "Wärmequelle": [
        (re.compile(r'(?i)luft-?wasser|\bluft\b'), 'Luft'),
        (re.compile(r'(?i)sole-?wasser|\bsole\b'), 'Sole'),
        (re.compile(r'(?i)brauchwasser'), 'Brauchwasser'),
        (re.compile(r'(?i)wasser-?wasser'), 'Wasser'),
    ],
    "Beschriftung": [
        (re.compile(r'(?i)dampf\s*12\s*bar'), 'Dampf 12 Bar'),
        (re.compile(r'(?i)dampf\s*8\s*bar'), 'Dampf 8 Bar'),
        (re.compile(r'(?i)dampf\s*3\s*bar'), 'Dampf 3 Bar'),
    ],
    "Antrieb": [
        (re.compile(r'(?i)riemenantrieb'), 'Riemenantrieb'),
        (re.compile(r'(?i)direktantrieb'), 'Direktantrieb'),
        (re.compile(r'(?i)(pozidriv|\bpz\b)'), 'Kreuzschlitz (Pozidriv)'),
        (re.compile(r'(?i)(\btx\d+\b|torx|t-profil|t-star|innensechsrund|\bi20\b|innenvielzahn)'), 'Torx'),
        (re.compile(r'(?i)(phillips|\bph\b|kreuzschlitz)'), 'Kreuzschlitz (Phillips)'),
        (re.compile(r'(?i)\bschlitz\b'), 'Schlitz'),
        (re.compile(r'(?i)6[,.]?3\s*mm\s*\(1/4''?\)\s*sechskant'), "6,3 mm (1/4'') Sechskant"),
        (re.compile(r'(?i)8\s*mm\s*\(5/16''?\)\s*sechskant'), "8 mm (5/16'') Sechskant"),
        (re.compile(r'(?i)9[,.]?5\s*mm\s*\(3/8''?\)\s*sechskant'), "9,5 mm (3/8'') Sechskant"),
        (re.compile(r'(?i)12[,.]?7\s*mm\s*\(1/2''?\)\s*vierkant'), "12,7 mm (1/2'') Vierkant"),
    ],
    "Schleifstoff": [
        (re.compile(r'(?i)(zirkon|zirc)'), 'Zirkonkorund'),
        (re.compile(r'(?i)(keramik|ceramic)'), 'Keramikkorn'),
        (re.compile(r'(?i)(siliciumcarbid|siliziumcarbid|sic)'), 'Siliciumcarbid'),
        (re.compile(r'(?i)aluminiumoxid'), 'Aluminiumoxid'),
        (re.compile(r'(?i)edelkorund'), 'Edelkorund'),
        (re.compile(r'(?i)korund'), 'Normalkorund'),
    ],
}
def normalize_text(text):
    text = str(text or "").lower()
    text = text.replace("×", "x").replace("∅", "ø")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_agg_values(raw_values):
    return re.findall(r"\[(.*?)\]", str(raw_values))


def normalize_numeric_string(number):
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def canonicalize_unit(unit):
    unit = str(unit or "").strip()
    return UNIT_CANONICAL.get(unit.lower(), unit)


def parse_numeric_value(value):
    value = str(value).strip().replace(",", ".")
    match = re.match(rf"^([0-9]+(?:\.[0-9]+)?)\s*({UNIT_PATTERN})?$", value)
    if not match:
        return None, None
    return float(match.group(1)), canonicalize_unit(match.group(2) or "")


def get_unit_family(unit):
    for family, mapping in UNIT_FACTORS.items():
        if unit in mapping:
            return family
    return None


def numeric_key(number, unit):
    family = get_unit_family(unit)
    if family:
        base_value = number * UNIT_FACTORS[family][unit]
        return family, round(base_value, 6)
    return unit, round(number, 6)


def compile_value_pattern(value, fuzzy=False):
    escaped = re.escape(value.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    if fuzzy:
        escaped = escaped.replace(r"\-", r"[-\s]?")
        escaped = escaped.replace(r"\(", r"(?:\(")
        escaped = escaped.replace(r"\)", r"\))?")
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(r"(?<![\w-])" + escaped + r"(?![\w-])", re.IGNORECASE)


def compile_numeric_value_pattern(value):
    value = str(value).strip().lower()
    escaped = re.escape(value)
    escaped = re.sub(r"\\[.,](?=\d)", r"[.,]", escaped)
    escaped = escaped.replace(r"\ ", r"\s*")
    escaped = escaped.replace(r"x", r"\s*x\s*")
    escaped = escaped.replace(r"\.", r"\.?")
    escaped = escaped.replace(r"\-", r"[-\s]?")
    escaped = escaped.replace(r"\+", r"\+?")
    escaped = escaped.replace(r"\(", r"\(\s*")
    escaped = escaped.replace(r"\)", r"\s*\)")
    escaped = escaped.replace(r"•", r"\s*•?")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


def canonicalize_text_match(text):
    return re.sub(r"\s+", " ", text.strip().lower())


@lru_cache(maxsize=None)
def get_feature_alias_part(feature_name):
    aliases = FEATURE_ALIASES.get(feature_name, [])
    return "|".join(re.escape(alias) for alias in aliases if alias)


@lru_cache(maxsize=None)
def get_dimension_alias_pattern(feature_name):
    alias_part = get_feature_alias_part(feature_name)
    if not alias_part:
        return None
    return re.compile(rf'(?i)(?:{alias_part})\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(µm|mm|cm|m|")\b')


@lru_cache(maxsize=None)
def get_numeric_anchored_pattern(feature_name, unit):
    alias_part = "|".join(re.escape(alias) for alias in FEATURE_ALIASES.get(feature_name, []) if len(alias) > 1)
    if not alias_part or not unit:
        return None
    return re.compile(rf"(?i)(?:{alias_part})\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*{re.escape(unit)}\b")


def earliest_pattern_match(pattern_value_pairs, text, allowed_values=None):
    hits = []
    for pattern, value in pattern_value_pairs:
        if allowed_values is not None and value not in allowed_values:
            continue
        match = pattern.search(text)
        if match:
            hits.append((match.start(), -(match.end() - match.start()), value))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def find_allowed_value(rule, *needles):
    lowered_needles = [needle.lower() for needle in needles if needle]
    for value in rule["values"]:
        lowered = value.lower()
        if all(needle in lowered for needle in lowered_needles):
            return value
    return None


def find_ral_value(rule, ral_code):
    if not ral_code:
        return None
    normalized = str(ral_code).replace("RAL", "").strip()
    for value in rule["values"]:
        if re.search(rf"(?i)ral\s*{re.escape(normalized)}\b", value):
            return value
    return None


def tokenize_alias_text(text):
    return [token for token in ALIAS_TOKEN_RE.findall(normalize_text(text)) if token not in ALIAS_STOPWORDS]


def should_keep_alias_phrase(phrase):
    if len(phrase) < 4:
        return False
    if phrase.isdigit():
        return False
    tokens = phrase.split()
    if not tokens:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    return True


def build_mined_categorical_aliases(train_alias_df):
    phrase_counts = defaultdict(Counter)
    for row in train_alias_df.itertuples(index=False):
        tokens = tokenize_alias_text(row.title)
        if not tokens:
            continue
        seen = set()
        max_n = min(4, len(tokens))
        for size in range(1, max_n + 1):
            for start in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[start:start + size])
                if phrase in seen or not should_keep_alias_phrase(phrase):
                    continue
                seen.add(phrase)
                phrase_counts[(row.category, row.feature_name, phrase)][row.feature_value] += 1

    alias_map = defaultdict(list)
    for (category, feature_name, phrase), counter in phrase_counts.items():
        total = sum(counter.values())
        value, support = counter.most_common(1)[0]
        precision = support / total
        if total < 3 or support < 3:
            continue
        if precision < 0.98:
            continue
        alias_map[(category, feature_name)].append((phrase, value, compile_value_pattern(phrase)))

    for key in alias_map:
        alias_map[key].sort(key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]))
    return dict(alias_map)


def extract_mined_alias_value(text, title_text, category, feature_name, rule, mined_aliases):
    entries = mined_aliases.get((category, feature_name), [])
    if not entries:
        return None
    for phrase, value, pattern in entries:
        if value not in rule["values"]:
            continue
        if pattern.search(title_text):
            return value
    for phrase, value, pattern in entries:
        if value not in rule["values"]:
            continue
        if pattern.search(text):
            return value
    return None


def choose_feature_mode(feature_name, allowed_values, feature_modes):
    for candidate, _ in feature_modes.get(feature_name, []):
        if candidate in allowed_values:
            return candidate
    return allowed_values[0] if allowed_values else None


def choose_category_feature_mode(category, feature_name, allowed_values, category_feature_modes, feature_modes):
    for candidate, _ in category_feature_modes.get((category, feature_name), []):
        if candidate in allowed_values:
            return candidate
    return choose_feature_mode(feature_name, allowed_values, feature_modes)


def choose_global_feature_mode(feature_name, feature_modes):
    ranked = feature_modes.get(feature_name, [])
    return ranked[0][0] if ranked else None


def extract_measurements(text):
    values = []
    for match in MEASUREMENT_RE.finditer(text):
        number = float(match.group("num").replace(",", "."))
        unit = canonicalize_unit(match.group("unit"))
        values.append((match.start(), number, unit))
    return values


def extract_tuple_candidates(text):
    tuples = []
    for pattern, labels in ORDERED_TUPLE_PATTERNS:
        for match in pattern.finditer(text):
            vals_match = COMMON_TUPLE_RE.search(match.group("vals"))
            if not vals_match:
                continue
            trail = vals_match.group("trail")
            slots = []
            for suffix in ["a", "b", "c", "d"][: len(labels)]:
                raw_num = vals_match.group(suffix)
                raw_unit = vals_match.group("u" + suffix) or trail
                if raw_num is None or raw_unit is None:
                    slots = []
                    break
                slots.append((float(raw_num.replace(",", ".")), canonicalize_unit(raw_unit)))
            if slots:
                tuples.append((match.start(), dict(zip(labels, slots))))
    compact = re.finditer(r"(?i)\b([hblt])\s*(\d+(?:[.,]\d+)?)\s*x\s*([btlh])\s*(\d+(?:[.,]\d+)?)\s*x\s*([hblt])\s*(\d+(?:[.,]\d+)?)(mm|cm|m)\b", text)
    for match in compact:
        labels = [match.group(1).upper(), match.group(3).upper(), match.group(5).upper()]
        unit = match.group(7)
        numbers = [
            float(match.group(2).replace(",", ".")),
            float(match.group(4).replace(",", ".")),
            float(match.group(6).replace(",", ".")),
        ]
        tuples.append((match.start(), dict(zip(labels, [(n, canonicalize_unit(unit)) for n in numbers]))))
    for match in GENERIC_DIMENSION_RE.finditer(text):
        trail = match.group("trail")
        slots = []
        for suffix in ["a", "b", "c", "d"]:
            raw_num = match.group(suffix)
            if raw_num is None:
                break
            raw_unit = match.group("u" + suffix) or trail
            slots.append((float(raw_num.replace(",", ".")), canonicalize_unit(raw_unit)))
        if len(slots) >= 2:
            tuples.append((match.start(), {"GENERIC": slots}))
    return tuples


def build_taxonomy_rules(tax_df):
    rules = {}
    for row in tax_df.itertuples(index=False):
        values = parse_agg_values(row.aggregated_feature_values)
        if not values:
            continue
        if row.feature_type == "categorical":
            fuzzy = row.feature_name in FUZZY_CATEGORICAL_FEATURES
            rules[(row.category, row.feature_name)] = {
                "type": "categorical",
                "values": sorted(values, key=len, reverse=True),
                "patterns": [(value, compile_value_pattern(value, fuzzy=fuzzy)) for value in sorted(values, key=len, reverse=True)],
                "fuzzy": fuzzy,
            }
            continue

        first_value = values[0]
        unit_number = parse_numeric_value(first_value)
        allowed_numeric = defaultdict(list)
        for value in values:
            number, unit = parse_numeric_value(value)
            if number is None:
                continue
            allowed_numeric[numeric_key(number, unit)].append(value)

        rules[(row.category, row.feature_name)] = {
            "type": "numeric",
            "values": values,
            "allowed_numeric": dict(allowed_numeric),
            "example": first_value,
            "parsed_example": unit_number,
            "category": row.category,
            "value_patterns": None,
            "use_value_patterns": unit_number[0] is None or row.feature_name in {"Format", "Schutzart", "Einzelzeichen", "Größe", "Verpackungseinheit"} or len(values) <= 40,
        }
    return rules


def build_feature_modes(train_feat_df):
    counters = defaultdict(Counter)
    for row in train_feat_df.itertuples(index=False):
        counters[row.feature_name][row.feature_value] += 1
    return {feature: counter.most_common(10) for feature, counter in counters.items()}


def build_category_feature_modes(train_feat_df):
    counters = defaultdict(Counter)
    for row in train_feat_df.itertuples(index=False):
        counters[(row.category, row.feature_name)][row.feature_value] += 1
    return {key: counter.most_common(10) for key, counter in counters.items()}


def lookup_numeric_candidate(rule, number, unit):
    key = numeric_key(number, unit)
    candidates = rule["allowed_numeric"].get(key)
    if candidates:
        return candidates[0]
    return None


def get_numeric_value_patterns(rule):
    if not rule.get("use_value_patterns"):
        return []
    if rule.get("value_patterns") is None:
        rule["value_patterns"] = [
            (value, compile_numeric_value_pattern(value))
            for value in sorted(rule["values"], key=len, reverse=True)
        ]
    return rule["value_patterns"]


def try_direct_numeric_values(text, rule):
    patterns = get_numeric_value_patterns(rule)
    for value, pattern in patterns:
        if pattern.search(text):
            return value
    return None


def extract_dimension_value(feature_name, text, measurements, tuple_candidates, rule):
    target_slot = DIMENSION_FEATURES.get(feature_name)
    if not target_slot:
        return None

    pattern = get_dimension_alias_pattern(feature_name)
    if pattern is not None:
        match = pattern.search(text)
        if match:
            candidate = lookup_numeric_candidate(rule, float(match.group(1).replace(",", ".")), match.group(2))
            if candidate:
                return candidate

    if feature_name == "Durchmesser":
        for match in DURCHMESSER_RE.finditer(text):
            candidate = lookup_numeric_candidate(rule, float(match.group(1).replace(",", ".")), match.group(2))
            if candidate:
                return candidate

    for pattern, index_map in COMPACT_DIMENSION_PATTERNS:
        match = pattern.search(text)
        if match and feature_name in index_map:
            index = index_map[feature_name]
            number = float(match.group(index + 1).replace(",", "."))
            unit = canonicalize_unit(match.group(4))
            candidate = lookup_numeric_candidate(rule, number, unit)
            if candidate:
                return candidate

    screw_pair = SCREW_PAIR_RE.search(text)
    if screw_pair:
        first = float(screw_pair.group(1).replace(",", "."))
        second = float(screw_pair.group(2).replace(",", "."))
        unit = canonicalize_unit(screw_pair.group(3) or (rule["parsed_example"][1] if rule["parsed_example"] else "mm"))
        if feature_name in {"Länge", "Gesamtlänge", "Kopflänge", "Spannutenlänge", "Ladeflächenlänge"}:
            candidate = lookup_numeric_candidate(rule, second, unit)
            if candidate:
                return candidate
        if feature_name in {"Durchmesser", "Nenn-Ø", "Rad-Ø", "Innen-Ø"}:
            candidate = lookup_numeric_candidate(rule, first, unit)
            if candidate:
                return candidate

    for _, tuple_values in tuple_candidates:
        keys = [target_slot]
        if target_slot == "D":
            keys.extend(["DI", "DA", "DK"])
        for key_name in keys:
            if key_name in tuple_values:
                number, unit = tuple_values[key_name]
                candidate = lookup_numeric_candidate(rule, number, unit)
                if candidate:
                    return candidate

        generic_slots = tuple_values.get("GENERIC")
        if generic_slots:
            example_unit = rule["parsed_example"][1] if rule["parsed_example"] else "mm"

            def generic_pick(index):
                if index >= len(generic_slots):
                    return None
                number, unit = generic_slots[index]
                unit = unit or example_unit
                return lookup_numeric_candidate(rule, number, unit)

            if feature_name in {"Durchmesser", "Nenn-Ø", "Rad-Ø"}:
                candidate = generic_pick(0)
                if candidate:
                    return candidate
            if feature_name == "Innen-Ø":
                candidate = generic_pick(0)
                if candidate:
                    return candidate
            if feature_name in {"Außen-Ø", "Kopf-Ø"}:
                candidate = generic_pick(1)
                if candidate:
                    return candidate
            if feature_name in {"Länge", "Gesamtlänge"}:
                candidate = generic_pick(1 if len(generic_slots) >= 2 else 0)
                if candidate:
                    return candidate
            if feature_name in {"Breite", "Gesamtbreite"}:
                candidate = generic_pick(0)
                if candidate:
                    return candidate
            if feature_name in {"Höhe", "Gesamthöhe"}:
                candidate = generic_pick(1 if len(generic_slots) >= 2 else 0)
                if candidate:
                    return candidate
            if feature_name in {"Tiefe", "Gesamttiefe"}:
                candidate = generic_pick(2 if len(generic_slots) >= 3 else 1)
                if candidate:
                    return candidate

    for _, number, unit in measurements:
        candidate = lookup_numeric_candidate(rule, number, unit)
        if candidate:
            return candidate

    return try_direct_numeric_values(text, rule)


def extract_packaging_value(text, rule):
    for pattern in PACKAGING_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        number = int(match.group(1).replace(".", ""))
        unit = match.group(2).lower() if len(match.groups()) > 1 else None
        if unit in {"st", "st.", "stück"}:
            candidate = f"{number} Stück"
            if candidate in rule["values"]:
                return candidate
        for value in rule["values"]:
            if re.match(rf"^{number}\s", value):
                return value
    return None


def extract_faecheranzahl_value(text, rule):
    for match in re.finditer(r"(?i)\b(\d+)\s*x\s*(\d+)\s*fächer\b", text):
        total = int(match.group(1)) * int(match.group(2))
        for value in rule["values"]:
            if value.startswith(f"{total} "):
                return value
    return None


def extract_range_value(feature_name, text, rule):
    range_match = None
    for pattern in RANGE_PATTERNS:
        range_match = pattern.search(text)
        if range_match:
            break
    if not range_match and feature_name in {"Messbereich von", "Messbereich bis", "min. Messbereich", "max. Messbereich"}:
        for pattern in DENSITY_RANGE_PATTERNS:
            range_match = pattern.search(text)
            if range_match:
                break
    if not range_match:
        return None
    low = float(range_match.group(1).replace(',', '.'))
    high = float(range_match.group(2).replace(',', '.'))
    unit = canonicalize_unit(range_match.group(3))
    target = None
    if feature_name in {"Spannbereich von", "Messbereich von", "min. Spannbereich", "min. Messbereich"}:
        target = low
    elif feature_name in {"Spannbereich bis", "Messbereich bis", "max. Spannbereich", "max. Messbereich"}:
        target = high
    if target is None:
        return None
    return lookup_numeric_candidate(rule, target, unit)


def extract_thread_value(feature_name, text, rule):
    if feature_name == "Antrieb":
        special_patterns = [
            (re.compile(r"(?i)riemenantrieb"), "Riemenantrieb"),
            (re.compile(r"(?i)direktantrieb"), "Direktantrieb"),
            (re.compile(r"(?i)(pozidriv|\bpz\b)"), "Kreuzschlitz (Pozidriv)"),
            (re.compile(r"(?i)(phillips|kreuzschlitz|\bph\b)"), "Kreuzschlitz (Phillips)"),
            (re.compile(r"(?i)(torx|t-profil|t-star|innensechsrund)"), "Torx"),
            (re.compile(r"(?i)\bschlitz\b"), "Schlitz"),
            (re.compile(r"(?i)6[,.]?3\s*mm\s*\(1/4''?\)\s*sechskant"), "6,3 mm (1/4'') Sechskant"),
            (re.compile(r"(?i)8\s*mm\s*\(5/16''?\)\s*sechskant"), "8 mm (5/16'') Sechskant"),
            (re.compile(r"(?i)9[,.]?5\s*mm\s*\(3/8''?\)\s*sechskant"), "9,5 mm (3/8'') Sechskant"),
            (re.compile(r"(?i)12[,.]?7\s*mm\s*\(1/2''?\)\s*vierkant"), "12,7 mm (1/2'') Vierkant"),
            (re.compile(r"(?i)innenvierkant"), "Innenvierkant"),
        ]
        for pattern, value in special_patterns:
            if value in rule["values"] and pattern.search(text):
                return value

    for pattern in THREAD_PATTERNS:
        for match in pattern.finditer(text):
            raw = re.sub(r"\s+", "", match.group(1).upper())
            for value in rule["values"]:
                if re.sub(r"\s+", "", value.upper()) == raw:
                    return value
    if feature_name == "Gewinde-Ø" and any(value.startswith("ST ") for value in rule["values"]):
        pair = re.search(r"(?i)(\d+(?:[.,]\d+)?)\s*x\s*\d+(?:[.,]\d+)?", text)
        if pair:
            candidate = f"ST {normalize_numeric_string(float(pair.group(1).replace(',', '.')))}"
            for value in rule["values"]:
                if value.replace(",", ".") == candidate.replace(",", "."):
                    return value
    return None


def extract_koernung_value(text, rule):
    for value in rule["values"]:
        escaped = re.escape(value)
        patterns = [
            re.compile(rf"(?i)körnung\s*(?:\(num\))?\s*{escaped}(?!\d)"),
            re.compile(rf"(?i)\bk\s*0*{escaped}(?!\d)"),
            re.compile(rf"(?i)\bp\s*0*{escaped}(?!\d)"),
            re.compile(rf"(?i)(?:schleif|körnung|co-cool|a\s){1,4}.*?\b{escaped}\b"),
        ]
        if any(pattern.search(text) for pattern in patterns):
            return value
    return None


def extract_luftdurchsatz_value(text, rule, measurements):
    ab_candidates = []
    for value in rule["values"]:
        match = re.match(r"(?i)ab\s+(\d+(?:[.,]\d+)?)\s*m³/h", value)
        if match:
            ab_candidates.append((float(match.group(1).replace(',', '.')), value))
    ab_candidates.sort()
    cpu_family = str(rule.get("category", "")).startswith("cpu_luefter")
    for _, number, unit in measurements:
        if unit.lower() != "m³/h":
            continue
        if cpu_family and 20 <= number <= 250:
            rounded_down = int(number // 10) * 10
            rounded_candidate = f"ab {rounded_down} m³/h"
            if rounded_candidate in rule["values"] and abs(number - rounded_down) <= 9:
                return rounded_candidate
        threshold_match = None
        for threshold, value in ab_candidates:
            if number >= threshold:
                threshold_match = value
            else:
                break
        if threshold_match:
            return threshold_match
        direct = lookup_numeric_candidate(rule, number, unit)
        if direct:
            return direct
        candidate = f"ab {normalize_numeric_string(number)} m³/h"
        if candidate in rule["values"]:
            return candidate
    for match in re.finditer(r"(?i)(\d+(?:[.,]\d+)?)\s*m³/h", text):
        candidate = f"ab {normalize_numeric_string(float(match.group(1).replace(',', '.')))} m³/h"
        if candidate in rule["values"]:
            return candidate
    return None


def extract_format_value(text, rule):
    for value in rule["values"]:
        if compile_value_pattern(value, fuzzy=True).search(text):
            return value
        din_match = re.match(r"(?i)(din\s+[a-z0-9]+)", value)
        if din_match and din_match.group(1).lower() in text:
            return value
    for match in FORMAT_DIMENSION_RE.finditer(text):
        first = normalize_numeric_string(float(match.group(1).replace(',', '.')))
        second = normalize_numeric_string(float(match.group(2).replace(',', '.')))
        unit = match.group(3)
        candidates = [
            f"{first} x {second} {unit}",
            f"{first}x{second} {unit}",
            f"{first} x {second}{unit}",
            f"{first}x{second}{unit}",
        ]
        for candidate in candidates:
            if candidate in rule["values"]:
                return candidate
    return None


def extract_bilddiagonale_value(text, rule):
    thresholds = []
    for value in rule["values"]:
        cm_match = re.search(r'(?i)ab\s+(\d+(?:[.,]\d+)?)\s*cm\s*\((\d+(?:[.,]\d+)?)\s*"\)', value)
        if cm_match:
            thresholds.append((float(cm_match.group(2).replace(',', '.')), value))
            continue
        inch_match = re.search(r'(?i)ab\s+(\d+(?:[.,]\d+)?)\s*(?:zoll|inch|")', value)
        if inch_match:
            thresholds.append((float(inch_match.group(1).replace(',', '.')), value))
    if not thresholds:
        return None
    thresholds.sort()

    explicit_inches = []
    for match in BILDDIAGONALE_INCH_RE.finditer(text):
        inches = float(match.group(1).replace(',', '.'))
        if 2 <= inches <= 100:
            explicit_inches.append(inches)
    converted_inches = []
    for match in BILDDIAGONALE_CM_RE.finditer(text):
        cm_value = float(match.group(1).replace(',', '.'))
        if 5 <= cm_value <= 300:
            converted_inches.append(cm_value / 2.54)

    observed_sets = []
    if explicit_inches:
        observed_sets.append((max(explicit_inches), 0.05))
    if converted_inches:
        observed_sets.append((max(converted_inches), 0.18))
    if not observed_sets:
        return None

    for observed, tolerance in observed_sets:
        chosen = None
        for threshold, value in thresholds:
            if observed + tolerance >= threshold:
                chosen = value
            else:
                break
        if chosen is not None:
            return chosen

    observed = max(value for value, _ in observed_sets)
    nearest = min(thresholds, key=lambda item: abs(item[0] - observed))
    if abs(nearest[0] - observed) <= 0.3:
        return nearest[1]
    return None


def extract_numeric_format_value(text, rule):
    for value, pattern in get_numeric_value_patterns(rule):
        if pattern.search(text):
            return value
    for match in FORMAT_DIMENSION_RE.finditer(text):
        first = normalize_numeric_string(float(match.group(1).replace(',', '.')))
        second = normalize_numeric_string(float(match.group(2).replace(',', '.')))
        unit = match.group(3)
        candidates = [
            f"{first} x {second} {unit}",
            f"{first}x{second} {unit}",
            f"{first} x {second}{unit}",
            f"{first}x{second}{unit}",
        ]
        for candidate in candidates:
            if candidate in rule["values"]:
                return candidate
    for match in FORMAT_PLAIN_RE.finditer(text):
        first = normalize_numeric_string(float(match.group(1).replace(',', '.')))
        second = normalize_numeric_string(float(match.group(2).replace(',', '.')))
        candidates = [
            f"{first} x {second} mm",
            f"{second} x {first} mm",
            f"{first} x {second}",
            f"{second} x {first}",
        ]
        for candidate in candidates:
            if candidate in rule["values"]:
                return candidate
    return None


def extract_special_categorical_value(text, title_text, rule, feature_name):
    combined_text = title_text + " " + text
    if feature_name == "Material":
        if re.search(r"(?i)fpm|viton|fluorkautschuk", combined_text):
            value = find_allowed_value(rule, "fkm") or find_allowed_value(rule, "fluorkautschuk")
            if value:
                return value
        if re.search(r"(?i)\bptfe\b|teflon|polytetrafluorethylen", combined_text):
            value = find_allowed_value(rule, "ptfe") or find_allowed_value(rule, "polytetrafluorethylen")
            if value:
                return value
        if re.search(r"(?i)\bpp\b|polypropylen", combined_text):
            value = find_allowed_value(rule, "pp") or find_allowed_value(rule, "polypropylen")
            if value:
                return value
        if re.search(r"(?i)\bpe\b|polyethylen", combined_text):
            value = find_allowed_value(rule, "pe") or find_allowed_value(rule, "polyethylen")
            if value:
                return value
        if re.search(r"(?i)federstahl|\bfst\b", combined_text):
            value = find_allowed_value(rule, "federstahl")
            if value:
                return value
        if re.search(r"(?i)1\.4310", combined_text):
            value = find_allowed_value(rule, "1.4310")
            if value:
                return value
        if re.search(r"(?i)4[,.]?8", combined_text):
            value = find_allowed_value(rule, "4.8") or find_allowed_value(rule, "4,8")
            if value:
                return value
        if re.search(r"(?i)4[,.]?6", combined_text):
            value = find_allowed_value(rule, "4.6") or find_allowed_value(rule, "4,6")
            if value:
                return value
        if re.search(r"(?i)kunstst\.?|kunststoff|nylon", combined_text):
            value = find_allowed_value(rule, "kunststoff")
            if value:
                return value
        if re.search(r"(?i)aluminium|\balu\b", combined_text):
            if re.search(r"(?i)kunststoffbeschichtet|beschichtet.*schwarz|schwarz.*beschichtet", combined_text):
                value = find_allowed_value(rule, "aluminium", "kunststoffbeschichtet", "schwarz")
                if value:
                    return value
            if re.search(r"(?i)blank", combined_text) and re.search(r"(?i)matt.*gestrahlt|gestrahlt.*matt|matt gestrahlt", combined_text):
                value = find_allowed_value(rule, "aluminium", "blank", "matt", "gestrahlt")
                if value:
                    return value
        checks = [
            (re.compile(r"(?i)stahl\s*verzinkt|galv(?:anisch)?\s*verzinkt"), "Stahl verzinkt"),
            (re.compile(r"(?i)edelstahl\s*a4|\ba4\b|1\.4401|1\.4571"), "Edelstahl (A4)"),
            (re.compile(r"(?i)edelstahl\s*a2|\ba2\b|1\.4301"), "Edelstahl (A2)"),
            (re.compile(r"(?i)edelstahl\s*a1|\ba1\b|1\.4305"), "Edelstahl (A1)"),
            (re.compile(r"(?i)\ba4\b"), "A4"),
            (re.compile(r"(?i)\ba2\b"), "A2"),
            (re.compile(r"(?i)\ba1\b"), "A1"),
            (re.compile(r"(?i)stahl\s*10[,.]?9|\b10[,.]?9\b"), "Stahl 10.9"),
            (re.compile(r"(?i)stahl\s*8[,.]?8|\b8[,.]?8\b"), "Stahl 8.8"),
            (re.compile(r"(?i)stahl\s*5[,.]?8|\b5[,.]?8\b"), "Stahl 5.8"),
            (re.compile(r"(?i)polyamid"), "Polyamid"),
            (re.compile(r"(?i)polypropylen"), "Polypropylen"),
            (re.compile(r"(?i)polycarbonat"), "Polycarbonat"),
            (re.compile(r"(?i)kunststoff|nylon"), "Kunststoff"),
            (re.compile(r"(?i)messing|\bms\b"), "Messing"),
            (re.compile(r"(?i)holz"), "Holz"),
            (re.compile(r"(?i)edelstahl"), "Edelstahl"),
            (re.compile(r"(?i)stahl|metall"), "Metall"),
            (re.compile(r"(?i)stahl"), "Stahl"),
            (re.compile(r"(?i)aluminium|\balu\b"), "Aluminium"),
        ]
        value = earliest_pattern_match(checks, combined_text, rule["values"])
        if value:
            return value
    if feature_name in {"Farbe", "Frontfarbe", "Korpusfarbe"}:
        if feature_name == "Korpusfarbe":
            explicit = re.search(r"(?i)korpus\s*ral\s*(\d{4})", combined_text)
            if explicit:
                value = find_ral_value(rule, explicit.group(1))
                if value:
                    return value
            slash_match = re.search(r"(?i)ral\s*(\d{4})\s*/\s*(\d{4})", combined_text)
            if slash_match:
                value = find_ral_value(rule, slash_match.group(1))
                if value:
                    return value
            standalone = re.search(r"(?i)\bral\s*(7035|7021|9010|9006)\b", combined_text)
            if standalone:
                value = find_ral_value(rule, standalone.group(1))
                if value:
                    return value
        if feature_name == "Frontfarbe":
            explicit = re.search(r"(?i)front\s*ral\s*(\d{4})", combined_text)
            if explicit:
                value = find_ral_value(rule, explicit.group(1))
                if value:
                    return value
            slash_match = re.search(r"(?i)ral\s*(\d{4})\s*/\s*(\d{4})", combined_text)
            if slash_match:
                value = find_ral_value(rule, slash_match.group(2))
                if value:
                    return value
        checks = [
            (re.compile(r"(?i)farblich sortiert|mehrfarbig|zubehör set"), "mehrfarbiges Set"),
            (re.compile(r"(?i)farblos|transparent|glasklar|clear\b"), "farblos (transparent)"),
            (re.compile(r"(?i)farblos|transparent|glasklar|clear\b"), "farblos"),
            (re.compile(r"(?i)\bcyan\b"), "cyan"),
            (re.compile(r"(?i)\borange\b"), "orange"),
            (re.compile(r"(?i)\bweiß\b|\bweiss\b|reinweiß|white\b"), "weiß"),
            (re.compile(r"(?i)\bschwarz\b|midnight|black\b"), "schwarz"),
            (re.compile(r"(?i)\bblau\b|royal\b|blue\b|lichtblau|enzianblau"), "blau"),
            (re.compile(r"(?i)\brosa\b|pink\b|rose\b|alpenglow"), "rosa"),
            (re.compile(r"(?i)\bgrau\b|grey\b|gray\b|veiled grey|db\s*703"), "grau"),
            (re.compile(r"(?i)\bgrün\b|gruen\b|green\b|sage green|resedagrün"), "grün"),
            (re.compile(r"(?i)\bviolett\b|lila\b|purple\b"), "violett"),
            (re.compile(r"(?i)\bgelb\b|yellow\b"), "gelb"),
            (re.compile(r"(?i)\brot\b|red\b"), "rot"),
            (re.compile(r"(?i)beige"), "beige"),
            (re.compile(r"(?i)terracotta"), "terracotta"),
        ]
        value = earliest_pattern_match(checks, combined_text, rule["values"])
        if value:
            return value
        ral_hits = []
        for value in rule["values"]:
            ral_match = re.search(r"(?i)(ral\s*\d{4})", value)
            if ral_match:
                found = re.search(ral_match.group(1), combined_text, re.IGNORECASE)
                if found:
                    ral_hits.append((found.start(), value))
        if ral_hits:
            ral_hits.sort()
            if feature_name == "Frontfarbe" and len(ral_hits) >= 2:
                return ral_hits[1][1]
            return ral_hits[0][1]
    if feature_name == "Oberfläche":
        checks = [
            (re.compile(r"(?i)galv(?:anisch)?\s*verzinkt"), "galvanisch verzinkt"),
            (re.compile(r"(?i)gelb\s*verzinkt|verzinkt\s*gelb"), "verzinkt gelb"),
            (re.compile(r"(?i)\bzink\b|\bverzinkt\b"), "verzinkt"),
            (re.compile(r"(?i)blank|edelstahl|\ba2\b|\ba4\b"), "blank"),
        ]
        for pattern, value in checks:
            if value in rule["values"] and pattern.search(text):
                return value
    if feature_name == "Ausführung":
        if "Standardhülle" in rule["values"] and re.search(r"(?i)zum\s+abheften|ringbuch|verschlussklapp|schutzvlies|für\s*\d+\s*cd", combined_text):
            return "Standardhülle"
        if "niedrige Form" in rule["values"] and re.search(r"(?i)\bniedrig\b", combined_text):
            return "niedrige Form"
        if "mittelhohe Form" in rule["values"] and re.search(r"(?i)mittelhoch", combined_text):
            return "mittelhohe Form"
        if "hohe Form" in rule["values"] and re.search(r"(?i)\bhoch\b", combined_text):
            return "hohe Form"
        checks = [
            (re.compile(r"(?i)flansch"), "Flanschverbindung"),
            (re.compile(r"(?i)reduzier|reduzierst(ü|u)ck"), "Reduzierstück"),
            (re.compile(r"(?i)falzdeckel"), "mit Falzdeckel"),
            (re.compile(r"(?i)bohrung|durchgangsbohrung"), "mit Bohrung"),
            (re.compile(r"(?i)stopfen"), "mit Stopfen"),
            (re.compile(r"(?i)rundhals"), "Rundhals"),
            (re.compile(r"(?i)\bmanuell\b|handbetrieb"), "manuell"),
            (re.compile(r"(?i)verschlu(?:ss|ß)kappe|\bkappe\b|endkappe"), "Endkappe"),
            (re.compile(r"(?i)deckenwinkel"), "Deckenwinkel"),
            (re.compile(r"(?i)schiebemuffe"), "Schiebemuffe"),
            (re.compile(r"(?i)winkelverschraubung"), "Winkelverschraubung"),
            (re.compile(r"(?i)\bt-?stück\b"), "T-Stück"),
            (re.compile(r"(?i)selbstklebetasche"), "Selbstklebetasche"),
            (re.compile(r"(?i)standardhülle"), "Standardhülle"),
            (re.compile(r"(?i)\bkombischild\b"), "Kombischild"),
            (re.compile(r"(?i)\bsymbol\s*schild\b|warnschild|gebotsschild|verbotsschild|hinweisschild"), "Symbol Schild"),
            (re.compile(r"(?i)\bwinkel\b"), "Winkel"),
            (re.compile(r"(?i)\bmuffe\b"), "Muffe"),
        ]
        for pattern, value in checks:
            if value in rule["values"] and pattern.search(combined_text):
                return value
    if feature_name == "Größe":
        if "Makro" in rule["values"] and re.search(r"(?i)\bmakro\b", text):
            return "Makro"
    if feature_name == "Form":
        if "Fußform" in rule["values"] and re.search(r"(?i)fuß-?form|fuß-verfahrschlitten", text):
            return "Fußform"
        if re.search(r"(?i)\b3\s*mm\s*flach\b", text):
            for value in rule["values"]:
                if "Flachform" in value and "3 mm" in value:
                    return value
        if re.search(r"(?i)\b2\s*mm\s*flach\b|\bflachform\b|\bflach\b", text):
            for value in rule["values"]:
                if "Flachform" in value:
                    return value
        if "Alform" in rule["values"] and re.search(r"(?i)\balform\b", text):
            return "Alform"
        if "Laschenform" in rule["values"] and re.search(r"(?i)laschenform", text):
            return "Laschenform"
    if feature_name == "Anschluss":
        checks = [
            (re.compile(r"(?i)mini\s*displayport"), "Mini Displayport"),
            (re.compile(r"(?i)displayport"), "Displayport"),
            (re.compile(r"(?i)\bdvi\b"), "DVI"),
            (re.compile(r"(?i)\busb\b"), "USB"),
            (re.compile(r"(?i)rs\s*422"), "RS422"),
            (re.compile(r"(?i)rs\s*232"), "RS232"),
            (re.compile(r"(?i)bluetooth"), "Bluetooth"),
        ]
        for pattern, value in checks:
            if value in rule["values"] and pattern.search(text):
                return value
    if feature_name == "Felgenmaterial":
        value = earliest_pattern_match([(re.compile(r"(?i)aluminium"), "Aluminium"), (re.compile(r"(?i)polypropylen|\btpa\b"), "Polypropylen"), (re.compile(r"(?i)polyamid|\bpath\b"), "Polyamid"), (re.compile(r"(?i)stahl"), "Stahl")], text, rule["values"])
        if value:
            return value
    if feature_name == "Laufbelag":
        value = earliest_pattern_match([(re.compile(r"(?i)thermoplast|\btpa\b"), "Thermoplast"), (re.compile(r"(?i)polyurethan|\bpath\b"), "Polyurethan"), (re.compile(r"(?i)gummi"), "Gummi"), (re.compile(r"(?i)kunststoff"), "Kunststoff")], text, rule["values"])
        if value:
            return value
    if feature_name == "Für Modell":
        value = earliest_pattern_match([(re.compile(r"(?i)raspberry\s*pi[^a-z0-9]{0,4}2\s*b"), "2B"), (re.compile(r"(?i)odroid\s*c2"), "C2"), (re.compile(r"(?i)odroid\s*c1\+"), "C1+"), (re.compile(r"(?i)raspberry\s*pi[^0-9]*5(?![0-9a-z])"), "5"), (re.compile(r"(?i)raspberry\s*pi[^0-9]*4(?![0-9a-z])"), "4"), (re.compile(r"(?i)raspberry\s*pi[^0-9]*3(?![0-9a-z])"), "3")], text, rule["values"])
        if value:
            return value
    if feature_name == "für Modelle von":
        value = earliest_pattern_match([(re.compile(r"(?i)zebra"), "Zebra"), (re.compile(r"(?i)honeywell"), "Honeywell"), (re.compile(r"(?i)mobile"), "Mobile")], text, rule["values"])
        if value:
            return value
    if feature_name == "Displayoberfläche":
        value = earliest_pattern_match([(re.compile(r"(?i)glänzend|glossy"), "glänzend"), (re.compile(r"(?i)matt|matte"), "matt")], text, rule["values"])
        if value:
            return value
    if feature_name == "Wärmequelle":
        checks = [
            (re.compile(r"(?i)luft-?wasser|\bluft\b"), "Luft"),
            (re.compile(r"(?i)sole-?wasser|\bsole\b"), "Sole"),
            (re.compile(r"(?i)brauchwasser"), "Brauchwasser"),
            (re.compile(r"(?i)wasser-?wasser"), "Wasser"),
        ]
        for pattern, value in checks:
            if value in rule["values"] and pattern.search(text):
                return value
    if feature_name == "Folie":
        for level in ["RA1", "RA2", "RA3"]:
            if level in rule["values"] and re.search(level.replace("RA", r"RA\s*"), text, re.IGNORECASE):
                return level
    if feature_name == "Dornmaterial":
        pair = re.search(r"(?i)(aluminium|stahl|edelstahl)\s*/\s*(aluminium|stahl|edelstahl)", text)
        if pair:
            second = pair.group(2).title()
            for value in rule["values"]:
                if value.lower().startswith(second.lower()):
                    return value
    if feature_name == "Brenngas":
        if "Acetylen, Propan" in rule["values"] and re.search(r"(?i)acetylen\s*/\s*propan", text):
            return "Acetylen, Propan"
    if feature_name == "Beschriftung":
        value = earliest_pattern_match([(re.compile(r"(?i)dampf\s*12\s*bar"), "Dampf 12 Bar"), (re.compile(r"(?i)dampf\s*8\s*bar"), "Dampf 8 Bar"), (re.compile(r"(?i)dampf\s*3\s*bar"), "Dampf 3 Bar")], text, rule["values"])
        if value:
            return value
    if feature_name == "Warenzustand":
        if "Wiederaufbereitet" in rule["values"] and re.search(r"(?i)refurbished|wiederaufbereitet|gebrauchtware", text):
            return "Wiederaufbereitet"
    if feature_name == "Antrieb":
        special_patterns = [
            (re.compile(r"(?i)riemenantrieb"), "Riemenantrieb"),
            (re.compile(r"(?i)direktantrieb"), "Direktantrieb"),
            (re.compile(r"(?i)(pozidriv|\bpz\b)"), "Kreuzschlitz (Pozidriv)"),
            (re.compile(r"(?i)(\btx\d+\b|torx|t-profil|t-star|innensechsrund|\bi20\b|innenvielzahn)"), "Torx"),
            (re.compile(r"(?i)(phillips|\bph\b|kreuzschlitz)"), "Kreuzschlitz (Phillips)"),
            (re.compile(r"(?i)\bschlitz\b"), "Schlitz"),
            (re.compile(r"(?i)6[,.]?3\s*mm\s*\(1/4''?\)\s*sechskant"), "6,3 mm (1/4'') Sechskant"),
            (re.compile(r"(?i)8\s*mm\s*\(5/16''?\)\s*sechskant"), "8 mm (5/16'') Sechskant"),
            (re.compile(r"(?i)9[,.]?5\s*mm\s*\(3/8''?\)\s*sechskant"), "9,5 mm (3/8'') Sechskant"),
            (re.compile(r"(?i)12[,.]?7\s*mm\s*\(1/2''?\)\s*vierkant"), "12,7 mm (1/2'') Vierkant"),
        ]
        for pattern, value in special_patterns:
            if value in rule["values"] and pattern.search(title_text + " " + text):
                return value
    if feature_name == "Säulentyp":
        if "präparativ" in text and "Präparativ" in rule["values"]:
            return "Präparativ"
        if ("hplc" in text or "trennsäule" in text or "vorsäule" in text) and "Analytisch" in rule["values"]:
            return "Analytisch"
    if feature_name == "Schleifstoff":
        checks = [
            (re.compile(r"(?i)(zirkon|zirc)"), "Zirkonkorund"),
            (re.compile(r"(?i)(keramik|ceramic)"), "Keramikkorn"),
            (re.compile(r"(?i)(siliciumcarbid|siliziumcarbid|sic)"), "Siliciumcarbid"),
            (re.compile(r"(?i)aluminiumoxid"), "Aluminiumoxid"),
            (re.compile(r"(?i)edelkorund"), "Edelkorund"),
            (re.compile(r"(?i)korund"), "Normalkorund"),
        ]
        for pattern, value in checks:
            if value in rule["values"] and pattern.search(text):
                return value
    return None
def extract_categorical_value(text, title_text, rule, category, feature_name, feature_modes, category_feature_modes, mined_aliases):
    if feature_name == "Format":
        format_value = extract_format_value(text, rule)
        if format_value:
            return format_value, "categorical_format"

    special_value = extract_special_categorical_value(text, title_text, rule, feature_name)
    if special_value:
        return special_value, "categorical_special"

    alias_value = extract_mined_alias_value(text, title_text, category, feature_name, rule, mined_aliases)
    if alias_value:
        return alias_value, "categorical_alias"

    matches = []
    title_matches = []
    for value, pattern in rule["patterns"]:
        if pattern.search(text):
            matches.append(value)
        if pattern.search(title_text):
            title_matches.append(value)

    if len(title_matches) == 1:
        return title_matches[0], "categorical_title_unique"
    if len(matches) == 1:
        return matches[0], "categorical_text_unique"
    if title_matches:
        if feature_name in {"Modell", "Material", "Beschriftung", "Brenngas", "für Modelle von", "Für Modell"}:
            positioned = []
            for value, pattern in rule["patterns"]:
                match = pattern.search(title_text)
                if match:
                    token_count = len(re.findall(r"[A-Za-z0-9]+", value))
                    positioned.append((-token_count, -(match.end() - match.start()), match.start(), value))
            if positioned:
                positioned.sort()
                return positioned[0][3], "categorical_title_multi"
        return choose_category_feature_mode(category, feature_name, title_matches, category_feature_modes, feature_modes), "categorical_title_multi"
    if matches:
        return choose_category_feature_mode(category, feature_name, matches, category_feature_modes, feature_modes), "categorical_text_multi"
    return choose_category_feature_mode(category, feature_name, rule["values"], category_feature_modes, feature_modes), "categorical_fallback"


def extract_numeric_value(feature_name, text, measurements, tuple_candidates, rule, feature_modes):
    if feature_name == "Format":
        value = extract_numeric_format_value(text, rule)
        if value:
            return value, "numeric_format"
    if feature_name in {"Gewinde-Ø", "Antrieb"}:
        value = extract_thread_value(feature_name, text, rule)
        if value:
            return value, "numeric_thread"
    if feature_name == "Verpackungseinheit":
        value = extract_packaging_value(text, rule)
        if value:
            return value, "numeric_packaging"
    if feature_name == "Bilddiagonale":
        value = extract_bilddiagonale_value(text, rule)
        if value:
            return value, "numeric_bilddiagonale"
    if feature_name == "Fächeranzahl":
        value = extract_faecheranzahl_value(text, rule)
        if value:
            return value, "numeric_faecher"
    if feature_name in {"Spannbereich von", "Messbereich von", "min. Spannbereich", "min. Messbereich", "Spannbereich bis", "Messbereich bis", "max. Spannbereich", "max. Messbereich"}:
        value = extract_range_value(feature_name, text, rule)
        if value:
            return value, "numeric_range"
    if feature_name == "Körnung":
        value = extract_koernung_value(text, rule)
        if value:
            return value, "numeric_koernung"
    if feature_name == "Luftdurchsatz":
        value = extract_luftdurchsatz_value(text, rule, measurements)
        if value:
            return value, "numeric_luftdurchsatz"
    if feature_name in DIMENSION_FEATURES:
        value = extract_dimension_value(feature_name, text, measurements, tuple_candidates, rule)
        if value:
            return value, "numeric_dimension"

    alias_part = "|".join(re.escape(alias) for alias in FEATURE_ALIASES.get(feature_name, []) if len(alias) > 1)
    if alias_part:
        unit = rule["parsed_example"][1] if rule["parsed_example"] else ""
        if unit:
            anchored = get_numeric_anchored_pattern(feature_name, unit)
            match = anchored.search(text) if anchored is not None else None
            if match:
                value = lookup_numeric_candidate(rule, float(match.group(1).replace(',', '.')), unit)
                if value:
                    return value, "numeric_anchored"

    direct = try_direct_numeric_values(text, rule)
    if direct:
        return direct, "numeric_direct"

    for _, number, unit in measurements:
        value = lookup_numeric_candidate(rule, number, unit)
        if value:
            return value, "numeric_measurement"

    return choose_feature_mode(feature_name, rule["values"], feature_modes), "numeric_fallback"


def build_predictions(prod_df, target_df, taxonomy_rules, feature_modes, category_feature_modes, mined_aliases):
    merged = target_df.merge(prod_df[["uid", "category", "title", "description"]], on="uid", how="left").reset_index().rename(columns={"index": "row_index"})
    predictions = [None] * len(merged)
    stages = [None] * len(merged)
    product_cache = {}
    process_df = pd.concat(
        [
            merged[merged["feature_type"] == "categorical"],
            merged[merged["feature_type"] != "categorical"],
        ],
        ignore_index=True,
    )

    for row in tqdm(process_df.itertuples(index=False), total=len(process_df)):
        rule = taxonomy_rules.get((row.category, row.feature_name))
        if rule is None:
            fallback_value = choose_global_feature_mode(row.feature_name, feature_modes)
            if row.feature_name == "Betriebsdauer (max.)":
                match = NO_RULE_BETRIEBSDAUER_RE.search(str(row.title) + " " + str(row.description))
                if match:
                    fallback_value = f"{normalize_numeric_string(float(match.group(1).replace(',', '.')))} min"
                    stages[row.row_index] = "no_rule_special"
                    predictions[row.row_index] = fallback_value
                    continue
            predictions[row.row_index] = fallback_value
            stages[row.row_index] = "no_rule_fallback"
            continue

        if row.uid not in product_cache:
            title_text = normalize_text(row.title)
            desc_text = normalize_text(row.description)
            full_text = (title_text + " " + desc_text).strip()
            product_cache[row.uid] = {
                "title": title_text,
                "text": full_text,
                "measurements": extract_measurements(full_text),
                "tuples": extract_tuple_candidates(full_text),
            }

        signals = product_cache[row.uid]
        if rule["type"] == "categorical":
            value, stage = extract_categorical_value(signals["text"], signals["title"], rule, row.category, row.feature_name, feature_modes, category_feature_modes, mined_aliases)
        else:
            value, stage = extract_numeric_value(row.feature_name, signals["text"], signals["measurements"], signals["tuples"], rule, feature_modes)
        predictions[row.row_index] = value
        stages[row.row_index] = stage

    return predictions, stages


def stage_sort_key(stage_name):
    if stage_name.startswith("categorical"):
        return (0, stage_name)
    if stage_name.startswith("numeric"):
        return (1, stage_name)
    return (2, stage_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tax_df = pd.read_parquet(os.path.join(DATA_ROOT, "taxonomy", "taxonomy.parquet"))
    taxonomy_rules = build_taxonomy_rules(tax_df)

    train_feat_df = pd.read_parquet(os.path.join(DATA_ROOT, "train", "product_features.parquet"), columns=["uid", "feature_name", "feature_value"])
    feature_modes = build_feature_modes(train_feat_df[["feature_name", "feature_value"]])
    train_prod_df = pd.read_parquet(os.path.join(DATA_ROOT, "train", "products.parquet"), columns=["uid", "category", "title", "description"])
    train_joined_df = train_feat_df.merge(train_prod_df, on="uid", how="left")
    category_feature_modes = build_category_feature_modes(train_joined_df[["category", "feature_name", "feature_value"]])
    mined_aliases = build_mined_categorical_aliases(train_joined_df[train_joined_df["feature_name"].isin(ALIAS_FEATURES)][["category", "feature_name", "feature_value", "title"]])

    prod_df = pd.read_parquet(os.path.join(DATA_ROOT, args.split, "products.parquet"))
    if args.split == "test":
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, "test", "submission.parquet"))
        truth = None
    else:
        target_df = pd.read_parquet(os.path.join(DATA_ROOT, "val", "product_features.parquet"))
        if args.sample_size:
            target_df = target_df.sample(min(args.sample_size, len(target_df)), random_state=args.seed).copy()
        truth = target_df["feature_value"].copy()
        target_df["feature_value"] = None

    predictions, stages = build_predictions(prod_df, target_df, taxonomy_rules, feature_modes, category_feature_modes, mined_aliases)
    target_df["feature_value"] = predictions
    target_df["prediction_stage"] = stages
    target_df[["uid", "feature_name", "feature_value", "feature_type"]].to_parquet(f"submission_{args.split}.parquet", index=False)
    print(f"Saved submission_{args.split}.parquet")

    if truth is not None:
        report_df = pd.DataFrame({
            "stage": stages,
            "pred": predictions,
            "truth": truth.values,
        })
        report_df["covered"] = report_df["pred"].notna()
        report_df["correct"] = report_df["pred"] == report_df["truth"]
        print(f"Overall coverage: {report_df['covered'].mean():.4f}")
        print(f"Overall accuracy: {report_df['correct'].mean():.4f}")
        ordered = sorted(set(stages), key=stage_sort_key)
        cumulative_mask = pd.Series(False, index=report_df.index)
        print("\nStage funnel:")
        for stage in ordered:
            stage_mask = report_df["stage"] == stage
            cumulative_mask |= stage_mask
            stage_count = int(stage_mask.sum())
            stage_acc = report_df.loc[stage_mask, "correct"].mean() if stage_count else 0.0
            cumulative_count = int(cumulative_mask.sum())
            cumulative_acc = report_df.loc[cumulative_mask, "correct"].mean() if cumulative_count else 0.0
            print(f"{stage:28s} count={stage_count:7d} stage_acc={stage_acc:.4f} cumulative={cumulative_count:7d} cumulative_acc={cumulative_acc:.4f}")


if __name__ == "__main__":
    main()
