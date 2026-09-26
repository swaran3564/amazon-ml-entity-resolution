"""Unit tests for the preprocessing and normalization module."""

import pytest
import pandas as pd
from src.preprocess import (
    normalize_name,
    compact_name,
    tokenized_name,
    normalize_address,
    compact_address,
    tokenized_address,
    extract_numeric_tokens,
    preprocess_dataframe,
)


# ============================================================================
# 1. Punctuation Variations
# ============================================================================

def test_name_punctuation_noise_brackets():
    """Verify bracket noise removal as documented in text_blocking_profile.md."""
    # Obsidian, LLC vs Obsidian, [[LLC]]
    assert normalize_name("Obsidian, LLC") == "obsidian llc"
    assert normalize_name("Obsidian, [[LLC]]") == "obsidian llc"
    assert normalize_name("Zander Blue Co") == "zander blue company"
    assert normalize_name("ZANDER BLUE [CO]") == "zander blue company"
    assert normalize_name("VADODARA INDUSTRIES PVT (LTD)") == "vadodara industries private limited"


def test_name_acronym_dots():
    """Verify period removal within acronyms (L.L.C. -> LLC, O.D. -> OD)."""
    assert normalize_name("Summit Health LLC") == "summit health llc"
    assert normalize_name("Summit Health L.L.C.") == "summit health llc"
    assert normalize_name("Renee G. Garcia, O.D.") == "renee g garcia od"
    assert normalize_name("Renee G Garcia, OD") == "renee g garcia od"
    assert normalize_name("D.D.S. Inc.") == "dds incorporated"


def test_name_ampersand_and_spacing():
    """Verify & to and conversion and whitespace normalization."""
    assert normalize_name("Brookens, Barrie & Smith") == "brookens barrie and smith"
    assert normalize_name("Brookens, Barrie &  Smith") == "brookens barrie and smith"
    assert normalize_name("Hendricks and  Flowers Inc") == "hendricks and flowers incorporated"


def test_noise_affixes():
    """Verify leading and trailing symbol noise is stripped."""
    assert normalize_name(">> caldeon nova") == "caldeon nova"
    assert normalize_name("-- Holloway Peak Inc") == "holloway peak incorporated"


# ============================================================================
# 2. Legal Suffix Variations
# ============================================================================

def test_legal_suffix_normalization():
    """Verify supported legal suffix variants normalize to canonical forms."""
    # ltd / limited
    assert normalize_name("Diksha Technologies Private Limited") == "diksha technologies private limited"
    assert normalize_name("Diksha Technologies Private Ltd") == "diksha technologies private limited"

    # pvt / private
    assert normalize_name("Style Multi Pvt. Ltd.") == "style multi private limited"
    assert normalize_name("Style Multi Private Limited") == "style multi private limited"

    # corp / corporation
    assert normalize_name("Dick Regional Armada Corp") == "dick regional armada corporation"
    assert normalize_name("[Corp] Dick Regional Armada") == "corporation dick regional armada"

    # inc / incorporated
    assert normalize_name("AP Hospitality Inc") == "ap hospitality incorporated"
    assert normalize_name("AP Hospitality Incorporated") == "ap hospitality incorporated"

    # co / company
    assert normalize_name("Zander Blue Co") == "zander blue company"
    assert normalize_name("Zander Blue Co.") == "zander blue company"
    # co-op must NOT be corrupted to company-op
    assert normalize_name("Co-op Services Inc") == "co-op services incorporated"

    # llc / llp
    assert normalize_name("Crystal Staffing Solutions LLC") == "crystal staffing solutions llc"
    assert normalize_name("Crystal Staffing Solutions L.L.C.") == "crystal staffing solutions llc"
    assert normalize_name("Express Care LLP") == "express care llp"


# ============================================================================
# 3. Word-Order Preservation
# ============================================================================

def test_word_order_preservation():
    """Verify that normalization preserves word order rather than sorting words."""
    name1 = "Hendricks and Flowers Inc"
    name2 = "Hendricks and Inc Flowers"
    assert tokenized_name(name1) == ["hendricks", "and", "flowers", "incorporated"]
    assert tokenized_name(name2) == ["hendricks", "and", "incorporated", "flowers"]

    rev1 = "Mumbai Producer Clinic"
    rev2 = "CLINIC PRODUCER MUMBAI"
    assert tokenized_name(rev1) == ["mumbai", "producer", "clinic"]
    assert tokenized_name(rev2) == ["clinic", "producer", "mumbai"]


# ============================================================================
# 4. Address Abbreviations
# ============================================================================

def test_address_abbreviations():
    """Verify street, road, drive, avenue, and Indian state abbreviation expansions."""
    # Street / st
    assert normalize_address("3315 Fremont Street, Peoria, IL") == "3315 fremont street peoria il"
    assert normalize_address("3315 FREMONT ST, PEORIA, IL") == "3315 fremont street peoria il"

    # Drive / dr
    assert normalize_address("8706 Kentucky Derby Drive, Waxhaw, NC") == "8706 kentucky derby drive waxhaw nc"
    assert normalize_address("8706 KENTUCKY DERBY DR, WAXHAW, NC") == "8706 kentucky derby drive waxhaw nc"

    # Indian state MH -> maharashtra, opp -> opposite, no -> number
    addr_short = "H No 847 S No 113/12 Opp Patil Enclave, Pune, MH"
    addr_long = "H No 847 S No 113/12 Opp Patil Enclave, Pune, Maharashtra"
    assert normalize_address(addr_short) == "h number 847 s number 113/12 opposite patil enclave pune maharashtra"
    assert normalize_address(addr_long) == "h number 847 s number 113/12 opposite patil enclave pune maharashtra"

    # Twp / township
    assert normalize_address("108 Norle Street, College Twp, PA") == "108 norle street college township pa"

    # Stn / station, nr / near
    assert "station" in normalize_address("5 Krishna Shopping Centre, Opp Dahisar Stn, P J Rd Dahisar E, Mumbai")
    assert "near" in normalize_address("Nr Metro Station, MG Rd, Bangalore")


# ============================================================================
# 5. Address Numbers
# ============================================================================

def test_extract_numeric_tokens():
    """Verify extraction of house numbers, ZIP codes, and PIN codes."""
    assert extract_numeric_tokens("3315 Fremont Street, Peoria, IL 61604") == ["3315", "61604"]
    assert extract_numeric_tokens("Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203") == [
        "684", "989", "9487203"
    ]
    assert extract_numeric_tokens("H No 847 S No 113/12") == ["847", "113", "12"]
    assert extract_numeric_tokens("No. 109, Ferns Meadows") == ["109"]
    assert extract_numeric_tokens("No numbers here") == []
    assert extract_numeric_tokens(None) == []


# ============================================================================
# 6. Missing Addresses and Null Safety
# ============================================================================

def test_missing_and_null_safety():
    """Verify robust handling of None, empty strings, and <NULL> artifacts."""
    for val in [None, "", "   ", "<NULL>", "<null>", "[null]"]:
        assert normalize_name(val) == ""
        assert compact_name(val) == ""
        assert tokenized_name(val) == []

        assert normalize_address(val) == ""
        assert compact_address(val) == ""
        assert tokenized_address(val) == []
        assert extract_numeric_tokens(val) == []

    # Address containing embedded <NULL> artifact
    raw_addr = "33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA"
    assert normalize_address(raw_addr) == "33466 warwick hills road yucaipa ca"


# ============================================================================
# 7. Alphanumeric Address Tokens
# ============================================================================

def test_alphanumeric_address_tokens_preserved():
    """Verify preservation of internal hyphens and slashes in alphanumeric tokens."""
    addr1 = "Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village"
    norm1 = normalize_address(addr1)
    assert "wz-187c" in norm1
    assert "s/f" in norm1

    addr2 = "Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1"
    norm2 = normalize_address(addr2)
    assert "e-3a" in norm2
    assert "g-1" in norm2

    addr3 = "D-4, Chandana Apartments82, Infantry Road"
    norm3 = normalize_address(addr3)
    assert "d-4" in norm3

    addr4 = "3/77, VIPUL KHAND"
    norm4 = normalize_address(addr4)
    assert "3/77" in norm4


# ============================================================================
# 8. Unicode and Non-ASCII Text
# ============================================================================

def test_unicode_and_non_ascii():
    """Verify proper handling of Devanagari and accented Latin scripts."""
    # Devanagari name with corporate suffixes
    dev_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
    norm_dev = normalize_name(dev_name)
    assert "राम" in norm_dev
    assert "मार्केटिंग" in norm_dev
    assert "private" in norm_dev
    assert "limited" in norm_dev

    # French accents
    fr_name = "Marina École France Sàrl"
    norm_fr = normalize_name(fr_name)
    assert norm_fr == "marina école france sàrl"

    # Hindi state in address
    hindi_addr = "Nandgram, Ghaziabad, उत्तर प्रदेश"
    norm_haddr = normalize_address(hindi_addr)
    assert "उत्तर प्रदेश" in norm_haddr


# ============================================================================
# 9. No Accidental Removal of Identity-Bearing Tokens
# ============================================================================

def test_no_accidental_removal_of_identity_bearing_tokens():
    """Ensure critical business identity words are never stripped."""
    check_names = [
        ("General Electric Company", ["general", "electric", "company"]),
        ("National Trust Bank", ["national", "trust", "bank"]),
        ("Central Medical Clinic", ["central", "medical", "clinic"]),
        ("American Red Cross", ["american", "red", "cross"]),
        ("Standard Chartered Private Bank", ["standard", "chartered", "private", "bank"]),
    ]
    for raw, expected_tokens in check_names:
        tokens = tokenized_name(raw)
        assert tokens == expected_tokens


# ============================================================================
# 10. DataFrame Helpers and Multiple Representations
# ============================================================================

def test_preprocess_dataframe_default_memory_conscious():
    """Verify default memory-conscious behavior does NOT materialize Python list columns."""
    df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["Obsidian, [[LLC]]", "Style Multi Pvt. Ltd."],
        "business_address": [
            "3315 FREMONT ST, PEORIA, IL",
            "H No 847 S No 113/12 Opp Patil Enclave, Pune, MH",
        ],
        "country": ["US", "India"],
    })

    df_out = preprocess_dataframe(df)

    # Verify original columns unchanged
    assert list(df["business_name"]) == ["Obsidian, [[LLC]]", "Style Multi Pvt. Ltd."]

    # Derived string columns present
    assert "name_normalized" in df_out.columns
    assert "name_compact" in df_out.columns
    assert "address_normalized" in df_out.columns
    assert "address_compact" in df_out.columns
    assert "country_normalized" in df_out.columns

    # Memory scalability check: token list columns MUST NOT be present by default
    assert "name_tokens" not in df_out.columns
    assert "address_tokens" not in df_out.columns
    assert "address_numeric_tokens" not in df_out.columns

    # Verify normalized string contents
    assert df_out["name_normalized"].iloc[0] == "obsidian llc"
    assert df_out["name_compact"].iloc[0] == "obsidianllc"
    assert df_out["address_normalized"].iloc[0] == "3315 fremont street peoria il"
    assert df_out["country_normalized"].iloc[0] == "US"


def test_preprocess_dataframe_optional_token_lists():
    """Verify optional token lists and numeric tokens are generated when requested."""
    df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["Obsidian, [[LLC]]", "Style Multi Pvt. Ltd."],
        "business_address": [
            "3315 FREMONT ST, PEORIA, IL",
            "H No 847 S No 113/12 Opp Patil Enclave, Pune, MH",
        ],
        "country": ["US", "India"],
    })

    df_out = preprocess_dataframe(df, include_token_lists=True)

    # All columns present when requested
    assert "name_tokens" in df_out.columns
    assert "address_tokens" in df_out.columns
    assert "address_numeric_tokens" in df_out.columns

    # Verify token contents
    assert df_out["name_tokens"].iloc[0] == ["obsidian", "llc"]
    assert df_out["address_tokens"].iloc[0] == ["3315", "fremont", "street", "peoria", "il"]
    assert df_out["address_numeric_tokens"].iloc[1] == ["847", "113", "12"]

