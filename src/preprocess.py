"""Data preprocessing and deterministic text normalization module.

Produces multiple specialized representations for business names and addresses
to feed candidate generation (blocking) and downstream feature engineering
in the locked architecture:
S1 -> (Char TF-IDF, Word TF-IDF, Embedding) -> Union -> All Features -> LightGBM -> F0.5
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd


# Precompiled regular expressions for performance on millions of records
NOISE_BRACKETS_RE = re.compile(r"[\[\]\(\)\{\}\<\>]")
NOISE_AFFIXES_RE = re.compile(r"^[\s\-_>>*#~]+|[\s\-_<<*#~]+$")
AMPERSAND_RE = re.compile(r"\s*&\s*")
ACRONYM_DOTS_RE = re.compile(
    r"\b([a-zA-Z])\.([a-zA-Z])(?:\.([a-zA-Z]))?(?:\.([a-zA-Z]))?\.?(?=\s|$|[^\w])"
)
PUNCT_NAME_CLEAN_RE = re.compile(r"[,;:\"\'\`!?~*#^+=|]")
TRAILING_DOT_RE = re.compile(r"\.(?=\s|$)")
MULTIPLE_SPACES_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")

# Legal suffix normalizations supported by observed training data
# Standardizes to full canonical forms while strictly preserving entity markers
LEGAL_SUFFIX_MAPPINGS = [
    (re.compile(r"\bpvt\.?\b", re.I), "private"),
    (re.compile(r"\bltd\.?\b", re.I), "limited"),
    (re.compile(r"\bcorp\.?\b|\bcorporation\b", re.I), "corporation"),
    (re.compile(r"\binc\.?\b|\bincorporated\b", re.I), "incorporated"),
    (re.compile(r"\bllc\.?\b", re.I), "llc"),
    (re.compile(r"\bllp\.?\b", re.I), "llp"),
    # Safe company mapping: standalone 'co' or 'co.' not followed by a hyphen or slash (preserves 'co-op')
    (re.compile(r"(?<!-)\bco\.(?=\s|$)|(?<!-)\bco\b(?!\s*[-/])", re.I), "company"),
]

# Devanagari corporate suffixes observed in Sources 2 & 3
DEVANAGARI_SUFFIX_MAPPINGS = [
    (re.compile(r"\bलिमिटेड\b"), "limited"),
    (re.compile(r"\bप्राइवेट\b"), "private"),
]

# Address abbreviation mappings supported by observed training data
ADDR_ABBREV_MAPPINGS = [
    (re.compile(r"\bst\b|\bst\.", re.I), "street"),
    (re.compile(r"\brd\b|\brd\.", re.I), "road"),
    (re.compile(r"\bdr\b|\bdr\.", re.I), "drive"),
    (re.compile(r"\bave\b|\bave\.", re.I), "avenue"),
    (re.compile(r"\bhwy\b|\bhwy\.", re.I), "highway"),
    (re.compile(r"\bapt\b|\bapt\.", re.I), "apartment"),
    (re.compile(r"\bno\b|\bno\.", re.I), "number"),
    (re.compile(r"\bopp\b|\bopp\.", re.I), "opposite"),
    (re.compile(r"\bnr\b|\bnr\.", re.I), "near"),
    (re.compile(r"\bstn\b|\bstn\.", re.I), "station"),
    (re.compile(r"\btwp\b|\btwp\.", re.I), "township"),
    (re.compile(r"\bblvd\b|\bblvd\.", re.I), "boulevard"),
    (re.compile(r"\bln\b|\bln\.", re.I), "lane"),
    (re.compile(r"\bste\b|\bste\.", re.I), "suite"),
    (re.compile(r"\bfl\b|\bfl\.", re.I), "floor"),
    (re.compile(r"\bbldg\b|\bbldg\.", re.I), "building"),
]

# Small explicit mapping for observed Indian state abbreviations
INDIAN_STATE_MAP = {
    "mh": "maharashtra",
    "dl": "delhi",
    "up": "uttar pradesh",
    "ka": "karnataka",
    "tn": "tamil nadu",
    "ts": "telangana",
    "ap": "andhra pradesh",
    "wb": "west bengal",
    "gj": "gujarat",
    "rj": "rajasthan",
    "hr": "haryana",
    "pb": "punjab",
    "mp": "madhya pradesh",
    "kl": "kerala",
    "br": "bihar",
    "jh": "jharkhand",
    "od": "odisha",
    "ch": "chandigarh",
    "uk": "uttarakhand",
    "ga": "goa",
}

INDIAN_STATE_PATTERNS = [
    (re.compile(rf"\b{abbr}\b", re.I), full)
    for abbr, full in INDIAN_STATE_MAP.items()
]

NULL_ARTIFACT_RE = re.compile(r"<null>|<nil>|\[null\]", re.I)
BRACKETS_ADDR_RE = re.compile(r"[,;:\"\'\`\(\)\[\]\{\}\<\>!?~*#^+=|]")
STANDALONE_HYPHEN_SLASH_RE = re.compile(r"\s+[-/]+\s+")
DANGLING_HYPHEN_SLASH_RE = re.compile(r"(?<=\s)[-/]+|[-/]+(?=\s)")
EDGE_HYPHEN_SLASH_RE = re.compile(r"^[-/]+|[-/]+$")


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a TSV dataset file with explicit tab separation."""
    return pd.read_csv(file_path, sep="\t", dtype=str)


def _remove_acronym_dots(text: str) -> str:
    """Remove periods within acronyms (e.g., 'L.L.C.' -> 'LLC', 'P.C.' -> 'PC')."""
    def repl(m):
        chars = [c for c in m.groups() if c is not None]
        return "".join(chars)
    return ACRONYM_DOTS_RE.sub(repl, text)


# ============================================================================
# 1. BUSINESS NAME NORMALIZATION
# ============================================================================

def normalize_name(name: Optional[str]) -> str:
    """Normalize a business name string.
    
    Operations:
    - Safe handling of null/empty/NaN values.
    - Unicode normalization (NFKC).
    - Lowercase conversion.
    - Surrounding noise/bracket removal ('[[LLC]]' -> 'LLC', '>>' -> '').
    - Standardizes '&' to 'and'.
    - Collapses acronym dots ('L.L.C.' -> 'llc').
    - Standardizes legal suffix variants (ltd/limited, pvt/private, corp/corporation, inc/incorporated, co/company).
    - Preserves all identity-bearing words and word ordering.
    """
    if name is None or pd.isna(name):
        return ""
    text = str(name).strip()
    if not text:
        return ""

    # Strip null tags
    text = NULL_ARTIFACT_RE.sub(" ", text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Strip leading/trailing symbol noise
    text = NOISE_AFFIXES_RE.sub("", text)

    # Remove noise brackets
    text = NOISE_BRACKETS_RE.sub(" ", text)

    # Standardize & to and
    text = AMPERSAND_RE.sub(" and ", text)

    # Remove acronym dots (e.g. L.L.C. -> LLC, O.D. -> OD)
    text = _remove_acronym_dots(text)

    # Lowercase
    text = text.lower()

    # Normalize legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_MAPPINGS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in DEVANAGARI_SUFFIX_MAPPINGS:
        text = pattern.sub(replacement, text)

    # Remove punctuation separators
    text = PUNCT_NAME_CLEAN_RE.sub(" ", text)

    # Remove trailing period from words
    text = TRAILING_DOT_RE.sub(" ", text)

    # Standardize whitespace
    text = MULTIPLE_SPACES_RE.sub(" ", text).strip()
    if text in ("null", "nil", "nan"):
        return ""
    return text


def compact_name(name: Optional[str]) -> str:
    """Produce a space-less compact representation of a business name.
    
    Useful for bridging token boundary inconsistencies (e.g., 'APHospitality' vs 'AP Hospitality').
    """
    normalized = normalize_name(name)
    if not normalized:
        return ""
    return normalized.replace(" ", "")


def tokenized_name(name: Optional[str]) -> List[str]:
    """Produce the ordered list of tokens from the normalized business name."""
    normalized = normalize_name(name)
    if not normalized:
        return []
    return normalized.split()


# ============================================================================
# 2. BUSINESS ADDRESS NORMALIZATION
# ============================================================================

def normalize_address(addr: Optional[str]) -> str:
    """Normalize a business address string.
    
    Operations:
    - Safe handling of null/empty/NaN values and '<NULL>' artifacts.
    - Unicode normalization (NFKC).
    - Lowercase conversion.
    - Preserves alphanumeric identifiers such as 'WZ-187C', 'E-3A', '113/12', 'G-1'.
    - Standardizes common street/unit abbreviations (st, rd, dr, ave, hwy, apt, no, opp, nr, stn, twp).
    - Standardizes observed Indian state abbreviations via small explicit dictionary (mh -> maharashtra, etc.).
    - Preserves landmarks ('near', 'opposite', 'behind', landmarks names).
    - Does NOT assume universal street-city-state ordering.
    """
    if addr is None or pd.isna(addr):
        return ""
    text = str(addr).strip()
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Strip null tags
    text = NULL_ARTIFACT_RE.sub(" ", text)

    # Lowercase
    text = text.lower()

    # Replace standalone hyphens or slashes surrounded by spaces with space
    text = STANDALONE_HYPHEN_SLASH_RE.sub(" ", text)

    # Remove separators and brackets, preserving internal hyphens/slashes in alphanumeric identifiers
    text = BRACKETS_ADDR_RE.sub(" ", text)

    # Replace periods with spaces (dots in addresses are abbreviation dots: 'rd.', 'st.', 'no.')
    text = text.replace(".", " ")

    # Replace & with and
    text = AMPERSAND_RE.sub(" and ", text)

    # Standardize street / address abbreviations
    for pattern, replacement in ADDR_ABBREV_MAPPINGS:
        text = pattern.sub(replacement, text)

    # Standardize observed Indian state abbreviations
    for pattern, replacement in INDIAN_STATE_PATTERNS:
        text = pattern.sub(replacement, text)

    # Clean dangling hyphens and slashes
    text = DANGLING_HYPHEN_SLASH_RE.sub(" ", text)
    text = EDGE_HYPHEN_SLASH_RE.sub(" ", text)

    # Standardize whitespace
    text = MULTIPLE_SPACES_RE.sub(" ", text).strip()
    if text in ("null", "nil", "nan"):
        return ""
    return text


def compact_address(addr: Optional[str]) -> str:
    """Produce a space-less compact representation of a business address."""
    normalized = normalize_address(addr)
    if not normalized:
        return ""
    return normalized.replace(" ", "")


def tokenized_address(addr: Optional[str]) -> List[str]:
    """Produce the ordered list of tokens from the normalized address."""
    normalized = normalize_address(addr)
    if not normalized:
        return []
    return normalized.split()


def extract_numeric_tokens(addr: Optional[str]) -> List[str]:
    """Extract all numeric sequences (house numbers, PIN codes, ZIP codes, plot numbers).
    
    Preserves exact sequence of digits found across the address string.
    """
    if addr is None or pd.isna(addr):
        return []
    text = str(addr).strip()
    if not text or text.lower() in ("<null>", "null", "nan"):
        return []
    return DIGITS_RE.findall(text)


# ============================================================================
# 3. SOURCE-AWARE DATAFRAME PREPROCESSING
# ============================================================================

def preprocess_dataframe(df: pd.DataFrame, include_token_lists: bool = False) -> pd.DataFrame:
    """Add multiple normalized representations to a business records DataFrame.
    
    Preserves original columns without unnecessary deep memory copies.
    
    Derived columns added by default (memory-conscious string representations):
    - If 'business_name' present:
      - 'name_normalized': canonical normalized name string
      - 'name_compact': whitespace-collapsed string
    - If 'business_address' present:
      - 'address_normalized': canonical normalized address string
      - 'address_compact': whitespace-collapsed address string
    - If 'country' present:
      - 'country_normalized': standardized uppercase country code
      
    Optional token list columns (enabled when include_token_lists=True for smaller batches):
    - 'name_tokens': list of token strings
    - 'address_tokens': list of address token strings
    - 'address_numeric_tokens': list of numeric tokens (house/ZIP/PIN numbers)
    """
    # Shallow copy preserves memory while permitting new column additions
    df_out = df.copy(deep=False)

    if "business_name" in df_out.columns:
        names = df_out["business_name"].tolist()
        norm_names = [normalize_name(x) for x in names]
        df_out["name_normalized"] = norm_names
        df_out["name_compact"] = [x.replace(" ", "") for x in norm_names]
        if include_token_lists:
            df_out["name_tokens"] = [x.split() for x in norm_names]

    if "business_address" in df_out.columns:
        addrs = df_out["business_address"].tolist()
        norm_addrs = [normalize_address(x) for x in addrs]
        df_out["address_normalized"] = norm_addrs
        df_out["address_compact"] = [x.replace(" ", "") for x in norm_addrs]
        if include_token_lists:
            df_out["address_tokens"] = [x.split() for x in norm_addrs]
            df_out["address_numeric_tokens"] = [extract_numeric_tokens(x) for x in addrs]

    if "country" in df_out.columns:
        df_out["country_normalized"] = [
            str(x).strip().upper() if (x is not None and pd.notna(x)) else "UNKNOWN"
            for x in df_out["country"]
        ]

    return df_out
