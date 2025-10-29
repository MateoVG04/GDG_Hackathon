import json
import os
from pathlib import Path
from ibm_watsonx_ai.foundation_models import ModelInference
import pandas as pd
from ibm_watsonx_ai import Credentials, APIClient

from scratch_files.gemini import Gemini

def setup_watsnox():
    api_key = os.getenv("WANSONX_API_KEY")
    project_id = "3a6f52db-292f-415d-a06b-47fb20378d26"
    if api_key is None:
        raise ValueError("API Key not provided")



    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=api_key,
    )
    client = APIClient(credentials)
    client.set.default_project(project_id) # Should print 'SUCCESS'

    foundational_model_id = "ibm/granite-3-2-8b-instruct"

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
              "'1:err1:err2:errX' DO NOT ANSWER ANYTHING ELSE APART FROM THIS FORMAT!!! Also don't write err1 or err2, this"
              "needs to be substituded by the actual error. Be as accurate"
              " as possible, because this is extreemly important!\n\n"
              +str(para_en) + "\n"+str(para_de) + "\n"+str(para_lv))

    result = LLM.generate(prompt).text
    print(result)
    return result

def parse_entire_text(data, LLM):
    paragraphs = len(data[0])
    df = pd.DataFrame(index=pd.Index([], dtype="Int64", name="para_number"),
                       columns=["flag", "errors"])
    df["flag"] = df["flag"].astype("boolean")

    for index in range(1, 14):
        para_en = data[0].loc[index]["para"]
        para_de = data[1].loc[index]["para"]
        para_lv = data[2].loc[index]["para"]

        para = (para_en, para_de, para_lv)
        result = find_errors(LLM, para)
        results = result.split(":")
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


if "__main__" == __name__:
    en_data, de_data, lv_data = load_json_test_samples()
    en_df = make_paragraph_df(en_data)
    de_df = make_paragraph_df(de_data)
    lv_df = make_paragraph_df(lv_data)

    prompt = "test prompt, say hi!"

    LLM = setup_watsnox()

    df = [en_df, de_df, lv_df]

    error_df = parse_entire_text(df, LLM)
    print(error_df)