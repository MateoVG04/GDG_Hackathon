import json
import re
from pathlib import Path

import pandas as pd

currency_codes = [
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL",
    "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY",
    "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP",
    "ERN", "ETB", "EUR", "FJD", "FKP", "FOK", "GBP", "GEL", "GGP", "GHS",
    "GIP", "GMD", "GNF", "GTQ", "GYD", "HKD", "HNL", "HRK", "HTG", "HUF",
    "IDR", "ILS", "IMP", "INR", "IQD", "IRR", "ISK", "JEP", "JMD", "JOD",
    "JPY", "KES", "KGS", "KHR", "KID", "KMF", "KRW", "KWD", "KYD", "KZT",
    "LAK", "LBP", "LKR", "LRD", "LSL", "LYD", "MAD", "MDL", "MGA", "MKD",
    "MMK", "MNT", "MOP", "MRU", "MUR", "MVR", "MWK", "MXN", "MYR", "MZN",
    "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB", "PEN", "PGK",
    "PHP", "PKR", "PLN", "PYG", "QAR", "RON", "RSD", "RUB", "RWF", "SAR",
    "SBD", "SCR", "SDG", "SEK", "SGD", "SHP", "SLE", "SOS", "SRD", "SSP",
    "STN", "SYP", "SZL", "THB", "TJS", "TMT", "TND", "TOP", "TRY", "TTD",
    "TVD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU", "UZS", "VES", "VND",
    "VUV", "WST", "XAF", "XCD", "XOF", "XPF", "YER", "ZAR", "ZMW", "ZWL"
]


def load_json_test_samples(new=False):
    if new:
        SCRIPT_DIR = Path(__file__).resolve().parent
        DATA_DIR = SCRIPT_DIR.parent / "new_data"
        en_path = DATA_DIR / "eval_sample_en.json"
        de_path = DATA_DIR / "eval_sample_de.json"
        lv_path = DATA_DIR / "eval_sample_lv.json"
    else:
        SCRIPT_DIR = Path(__file__).resolve().parent
        DATA_DIR = SCRIPT_DIR.parent / "data"
        en_path = DATA_DIR / "test_sample_en_parsed.json"
        de_path = DATA_DIR / "test_sample_de_parsed.json"
        lv_path = DATA_DIR / "test_sample_lv_parsed.json"

    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)
    with open(de_path, "r", encoding="utf-8") as f:
        de_data = json.load(f)
    with open(lv_path, "r", encoding="utf-8") as f:
        lv_data = json.load(f)

    return en_data, de_data, lv_data

def make_paragraph_df(data):
    """
    data = list of documents, each with {"file": ..., "para": [{"para_number": int, "para": str}, ...]}
    Returns a DataFrame with columns: file, para_number, para
    Index is para_number (not necessarily unique across files).
    """
    rows = []
    for doc in data:
        file = doc.get("file")
        for p in doc.get("para", []):
            rows.append({
                "file": file,
                "para_number": p.get("para_number"),
                "para": (p.get("para") or "").rstrip("\n"),
            })
    df = pd.DataFrame(rows)
    # keep only rows with a number, sort, and set an index you can work with
    df = df.dropna(subset=["para_number"]).sort_values(["file", "para_number"])
    return df.set_index("para_number", drop=False)


LANGS = ["en", "de", "lv"]

def _count_codes(text: str, codes: list[str]) -> dict[str, int]:
    """Count ISO currency codes as whole words (case-sensitive)."""
    out = {}
    for code in codes:
        n = len(re.findall(rf"\b{re.escape(code)}\b", text))
        if n:
            out[code] = n
    return out

def compare_currency_counts(dfs: list[pd.DataFrame], codes: list[str]) -> pd.DataFrame:
    """
    dfs = [en_df, de_df, lv_df] (all indexed by para_number and with column 'para')
    Returns a DF with columns:
      - para_number
      - has_mismatch (bool)
      - mismatches (dict: {code: {'en': x, 'de': y, 'lv': z}})
    """
    common_idx = dfs[0].index.intersection(dfs[1].index).intersection(dfs[2].index)
    rows = []

    for i in common_idx:
        texts = {
            "en": dfs[0].loc[i, "para"],
            "de": dfs[1].loc[i, "para"],
            "lv": dfs[2].loc[i, "para"],
        }

        counts = {lang: _count_codes(texts[lang], codes) for lang in LANGS}
        codes_here = set().union(*[set(d.keys()) for d in counts.values()])

        mismatches = {}
        for code in sorted(codes_here):
            triple = (counts["en"].get(code, 0),
                      counts["de"].get(code, 0),
                      counts["lv"].get(code, 0))
            if not (triple[0] == triple[1] == triple[2]):
                mismatches[code] = {"en": triple[0], "de": triple[1], "lv": triple[2]}

        rows.append({
            "para_number": i,
            "has_mismatch": bool(mismatches),
            "mismatches": mismatches
        })

    out = pd.DataFrame(rows).set_index("para_number")
    return out

import matplotlib.pyplot as plt

# def plot_currency_counts(counts: dict[str, dict[str, int]], save=True, show=True):
#     """
#     counts example:
#     {
#         "EUR": {"en": 3, "de": 2, "lv": 3},
#         "USD": {"en": 0, "de": 1, "lv": 0},
#         ...
#     }
#     Makes one bar chart per currency with three bars (en, de, lv).
#     """
#     languages = ["en", "de", "lv"]
#
#     for cur in sorted(counts.keys()):
#         vals = [counts[cur].get(lang, 0) for lang in languages]
#
#         plt.figure(figsize=(4.5, 3.2))
#         x = range(len(languages))
#         plt.bar(x, vals)
#         plt.xticks(x, languages)
#         plt.title(f"{cur} occurrences by language")
#         plt.ylabel("count")
#         plt.tight_layout()
#
#         if save:
#             plt.savefig(f"{cur}_counts.png", dpi=150)
#         if show:
#             plt.show()
#         else:
#             plt.close()

import numpy as np
def plot_currency_counts(counts: dict[str, dict[str, int]], save=None, show=True):
    """
    counts example:
    {
        "EUR": {"en": 3, "de": 2, "lv": 3},
        "USD": {"en": 0, "de": 1, "lv": 0},
        ...
    }
    Plots one grouped bar chart with three bars per currency (en, de, lv).
    """
    languages = ["en", "de", "lv"]
    currencies = sorted(counts.keys())

    # Build a matrix of shape (len(languages), len(currencies))
    vals = np.array([[counts[cur].get(lang, 0) for cur in currencies] for lang in languages])

    x = np.arange(len(currencies))            # group positions
    width = 0.8 / len(languages)              # bar width so groups fit nicely

    plt.figure(figsize=(max(6, 1.2 * len(currencies)), 4.0))
    for i, lang in enumerate(languages):
        plt.bar(x + i * width - (width*(len(languages)-1)/2), vals[i], width, label=lang)

    plt.xticks(x, currencies, rotation=45, ha="right")
    plt.ylabel("count")
    plt.title("Currency mentions by language")
    plt.legend(title="language")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

if "__main__" == __name__:
    en_data, de_data, lv_data = load_json_test_samples(new=True)
    en_df = make_paragraph_df(en_data)
    de_df = make_paragraph_df(de_data)
    lv_df = make_paragraph_df(lv_data)

    result = compare_currency_counts([en_df, de_df, lv_df], currency_codes)
    print(result)
    print("result2")
    print(result[result["has_mismatch"]])
    candidates = result[result["has_mismatch"]]
    #real_errors = filter_out_false_positives(candidates, currency_codes)

    print("result3")
    print(result.loc[13, "mismatches"])
    plot_currency_counts(result.loc[13, "mismatches"], save=False)