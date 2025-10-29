import json
import os
from pathlib import Path
from ibm_watsonx_ai.foundation_models import ModelInference
import pandas as pd
from ibm_watsonx_ai import Credentials, APIClient
from dotenv import load_dotenv, find_dotenv
import re
from collections import Counter


from scratch_files.gemini import Gemini

# parameters = {
#     "decoding_method": "greedy",
#     "temperature": 0,
#     "max_new_tokens": 500,
#     "response_format": {
#         "type": "json",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "answer": {"type": "string"},
#                 "confidence": {"type": "number"},
#                 "sources": {"type": "array", "items": {"type": "string"}}
#             },
#             "required": ["answer", "confidence"]
#         }
#     }
# }

parameters = {
    "decoding_method": "greedy",
    "temperature": 0,
    "max_new_tokens": 16,
    "min_new_tokens": 0,
    "stop_sequences": ["\n"],
    "response_format": {
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer", "confidence"]
        }
    }
}

currency_symbols = [
    "$",   # Dollar (USD, CAD, AUD, etc.)
    "€",   # Euro
    "¥",   # Yen/Yuan
    "£",   # Pound
    "₣",   # Franc (historic, still used in some)
    "₩",   # South Korean Won
    "₹",   # Indian Rupee
    "₽",   # Russian Ruble
    "R$",  # Brazilian Real
    "R",   # South African Rand
    "₺",   # Turkish Lira
    "₪",   # Israeli Shekel
    "﷼",   # Generic Rial/Riyal (SAR, QAR, OMR, IRR, YER)
    "د.إ", # UAE Dirham
    "₱",   # Philippine Peso
    "₫",   # Vietnamese Dong
    "₦",   # Nigerian Naira
    "د.ج", # Algerian Dinar
    "د.ك", # Kuwaiti Dinar
    "ج.م", # Egyptian Pound
    "د.ت", # Tunisian Dinar
    "ر.ق", # Qatari Riyal
    "د.ب", # Bahraini Dinar
    "ل.ل", # Lebanese Pound
    "฿",   # Thai Baht
    "₭",   # Lao Kip
    "₮",   # Mongolian Tögrög
    "₴",   # Ukrainian Hryvnia
    "лв",  # Bulgarian Lev
    "Ft",  # Hungarian Forint
    "zł",  # Polish Zloty
    "kr",  # Nordic Krona/Krone (SEK, NOK, DKK, ISK)
    "Kč",  # Czech Koruna
    "lei", # Romanian Leu
    "DH",  # Moroccan Dirham (common abbreviation)
    "₲",   # Paraguayan Guaraní
    "₡",   # Costa Rican Colón
    "₵",   # Ghanaian Cedi
    "₸",   # Kazakhstani Tenge
    "₼",   # Azerbaijani Manat
    "ман", # Turkmenistani Manat
    "ден", # Macedonian Denar
    "؋",   # Afghan Afghani
    "Br",  # Ethiopian Birr / Belarusian Ruble
    "₨",   # Generic Rupee (Sri Lanka, Nepal, Pakistan, etc.)
]


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


currency_names = [
    "united arab emirates dirham",
    "afghan afghani",
    "albanian lek",
    "armenian dram",
    "netherlands antillean guilder",
    "angolan kwanza",
    "argentine peso",
    "australian dollar",
    "aruban florin",
    "azerbaijani manat",
    "bosnia and herzegovina convertible mark",
    "barbados dollar",
    "bangladeshi taka",
    "bulgarian lev",
    "bahraini dinar",
    "burundian franc",
    "bermudian dollar",
    "brunei dollar",
    "bolivian boliviano",
    "brazilian real",
    "bahamian dollar",
    "bhutanese ngultrum",
    "botswana pula",
    "belarusian ruble",
    "belize dollar",
    "canadian dollar",
    "congolese franc",
    "swiss franc",
    "chilean peso",
    "chinese yuan renminbi",
    "colombian peso",
    "costa rican colón",
    "cuban peso",
    "cape verdean escudo",
    "czech koruna",
    "djiboutian franc",
    "danish krone",
    "dominican peso",
    "algerian dinar",
    "egyptian pound",
    "eritrean nakfa",
    "ethiopian birr",
    "euro",
    "fiji dollar",
    "falkland islands pound",
    "faroese krona",
    "pound sterling",
    "georgian lari",
    "guernsey pound",
    "ghanaian cedi",
    "gibraltar pound",
    "gambian dalasi",
    "guinean franc",
    "guatemalan quetzal",
    "guyana dollar",
    "hong kong dollar",
    "honduran lempira",
    "croatian kuna",     # replaced by euro in 2023, still ISO
    "haitian gourde",
    "hungarian forint",
    "indonesian rupiah",
    "israeli new shekel",
    "jersey pound",
    "indian rupee",
    "iraqi dinar",
    "iranian rial",
    "icelandic krona",
    "jersey pound",
    "jamaican dollar",
    "jordanian dinar",
    "japanese yen",
    "kenyan shilling",
    "kyrgyzstani som",
    "cambodian riel",
    "kiribati dollar",
    "comorian franc",
    "south korean won",
    "kuwaiti dinar",
    "cayman islands dollar",
    "kazakhstani tenge",
    "lao kip",
    "lebanese pound",
    "sri lanka rupee",
    "liberian dollar",
    "lesotho loti",
    "libyan dinar",
    "moroccan dirham",
    "moldovan leu",
    "malagasy ariary",
    "macedonian denar",
    "myanmar kyat",
    "mongolian tögrög",
    "macanese pataca",
    "mauritanian ouguiya",
    "mauritian rupee",
    "maldivian rufiyaa",
    "malawian kwacha",
    "mexican peso",
    "malaysian ringgit",
    "mozambican metical",
    "namibian dollar",
    "nigerian naira",
    "nicaraguan córdoba",
    "norwegian krone",
    "nepalese rupee",
    "new zealand dollar",
    "omani rial",
    "panamanian balboa",
    "peruvian sol",
    "papua new guinea kina",
    "philippine peso",
    "pakistani rupee",
    "polish złoty",
    "paraguayan guaraní",
    "qatari riyal",
    "romanian leu",
    "serbian dinar",
    "russian ruble",
    "rwandan franc",
    "saudi riyal",
    "solomon islands dollar",
    "seychelles rupee",
    "sudanese pound",
    "swedish krona",
    "singapore dollar",
    "saint helena pound",
    "sierra leonean leone",
    "somali shilling",
    "surinamese dollar",
    "south sudanese pound",
    "são tomé and príncipe dobra",
    "syrian pound",
    "eswatini lilangeni",
    "thai baht",
    "tajikistani somoni",
    "turkmenistani manat",
    "tunisian dinar",
    "tongan paʻanga",
    "turkish lira",
    "trinidad and tobago dollar",
    "tuvaluan dollar",
    "new taiwan dollar",
    "tanzanian shilling",
    "ukrainian hryvnia",
    "ugandan shilling",
    "united states dollar",
    "uruguayan peso",
    "uzbekistani som",
    "venezuelan bolívar",
    "vietnamese đồng",
    "vanuatu vatu",
    "samoan tālā",
    "central african cfa franc",
    "east caribbean dollar",
    "west african cfa franc",
    "cfp franc",
    "yemeni rial",
    "south african rand",
    "zambian kwacha",
    "zimbabwean dollar"
]



def setup_watsnox():
    loaded = load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv("api_key")
    project_id = "3a6f52db-292f-415d-a06b-47fb20378d26"
    if api_key is None:
        raise ValueError("API Key not provided")

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=api_key,
    )
    client = APIClient(credentials)
    client.set.default_project(project_id) # Should print 'SUCCESS'

    #foundational_model_id = "openai/gpt-oss-120b"
    foundational_model_id = "openai/gpt-oss-120b"


    foundational_model = ModelInference(
        model_id=foundational_model_id,
        credentials=credentials,
        project_id=project_id
    )
    return foundational_model


def load_json_test_samples():
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

def find_errors(LLM, para):
    para_en, para_de, para_lv = para

    prompt = ("You will be given 3 paragraphs in different languages. The text should be exactly the same. However,"
              "sometimes errors occur. Check the paragraph below and look specifically for the numbers and dates."
              "See if there are errors/mismatches between the paragraphs. You need to return 1 or 0, when there is a "
              "error/mismatch then return 1 otherwise 0. Also add what is wrong! You should follow this format:"
              "'1:err1:err2:errX' DO NOT ANSWER ANYTHING ELSE APART FROM THIS FORMAT!!! Also don't write err1 or err2, this "
              "needs to be substituded by the actual error. Don't give an explanation, only give the format nothing more nothing"
              "less! Be as accurate"
              " as possible, because this is extreemly important!\n\n"
              +str(para_en) + "\n\n"+str(para_de) + "\n\n"+str(para_lv))

    #prompt = ("Is in this paragraph some kinde of mention of valuta? Only answer with Yes or No")


    result = LLM.generate(prompt=prompt,params=parameters)
    print(result)
    text = result["results"][0]["generated_text"].strip()
    results = text.split("\n")
    results = [r for r in results if r.strip()]

    return results

def parse_entire_text(data, LLM):
    paragraphs = len(data[0])
    df = pd.DataFrame(index=pd.Index([], dtype="Int64", name="para_number"),
                       columns=["flag", "errors"])
    df["flag"] = df["flag"].astype("boolean")

    for index in range(1, paragraphs):
        para_en = data[0].loc[index]["para"]
        para_de = data[1].loc[index]["para"]
        para_lv = data[2].loc[index]["para"]

        para = (para_en, para_de, para_lv)

        result = find_errors(LLM, para)
        if result is not None:
            results = result[0].split(":")
            #extra_info = results[1]

            print(results)
            df.at[index, "flag"] = True
            if results[0] == "1":
                results.pop(0)
                print(results)
                df.at[index,"flag"] = True
                df.at[index,"errors"] = results

            else:
                df.loc[index, "flag"] = False
    return df

def valuta_counter(data):
    counts = Counter()
    paragraphs = len(data[0])
    for index in range(1, 5):
        for language in data:
            para_language = language.loc[index]["para"]
            print(para_language+"\n")

            for sym in currency_symbols:
                if sym in para_language:
                    counts[sym] = para_language.count(sym)

                # Abbreviations (whole word, case-sensitive like USD/EUR)
            for code in currency_codes:
                matches = re.findall(rf"\b{re.escape(code)}\b", para_language)
                if matches:
                    counts[code] = len(matches)

                # Full names (whole phrase, case-insensitive)
            for name in currency_names:
                matches = re.findall(rf"\b{name}\b", para_language, flags=re.IGNORECASE)
                if matches:
                    counts[name] = len(matches)

    return counts


if "__main__" == __name__:
    en_data, de_data, lv_data = load_json_test_samples()
    en_df = make_paragraph_df(en_data)
    de_df = make_paragraph_df(de_data)
    lv_df = make_paragraph_df(lv_data)

    prompt = "test prompt, say hi!"

    #LLM = setup_watsnox()

    df = [en_df, de_df, lv_df]

    error_df = valuta_counter(df)
    print(error_df)