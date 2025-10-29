import json
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

# Load paragraph from json file
def read_paragraphs_from_json(json_file):
    # Open and load the JSON file
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data[0]["para"]

# EN
paragraphs_en = read_paragraphs_from_json("test_sample_en_parsed.json")
# LV
paragraphs_lv = read_paragraphs_from_json("test_sample_lv_parsed.json")


# Split the paragraph into words and assign them an index
# Next extract all numbers, if there occur mulitple in the same word -> extract seperatly but assing the same word index
# as they appear in the same word
def get_all_strings_containing_numbers(paragraph):
    words = paragraph.replace(" %","%").split()
    number_words=[]
    indexes=[]
    for i,word in enumerate(words):
        # print(word)
        tokens = re.findall(r'\b\d+%|\b\w+\b', word)

        # tokens = re.findall(r'\b\w+\b', word)
        # print("tokens",tokens)
        # print(tokens)
        for token in tokens:
            # print(token)
            if re.search(r'\d', token):
                number_words.append(token)
                indexes.append(i)
    return number_words, indexes

def clean_token(token):
    return token


# Compute a fuzzy numeric similarity between two strings.
def numeric_similarity(a, b):
    if not a or not b:
        return 0.0
    a_norm, b_norm = clean_token(a), clean_token(b)
    if a_norm == b_norm:
        return 1.0
    # Overlap of numbers
    a_nums, b_nums = set(a_norm.split()), set(b_norm.split())
    if not a_nums or not b_nums:
        return 0.0
    inter = len(a_nums & b_nums)
    union = len(a_nums | b_nums)
    return inter / union  # simple Jaccard similarity

# Align vectors:
# if not the same length -> pad the shortest
def semantic_align(seq1, seq2, indexes1, indexes2, placeholder=None):
    n, m = len(seq1), len(seq2)
    size = max(n, m)
    cost = np.zeros((size, size))

    # Build similarity matrix (1 - similarity = cost)
    for i in range(size):
        for j in range(size):
            if i < n and j < m:
                sim = numeric_similarity(seq1[i], seq2[j])
                cost[i, j] = 1 - sim
            else:
                cost[i, j] = 1  # gap penalty

    # Hungarian algorithm to minimize total cost (maximize similarity)
    row_ind, col_ind = linear_sum_assignment(cost)

    aligned1, aligned2 = [], []
    aligned_idx1, aligned_idx2 = [], []

    for i, j in zip(row_ind, col_ind):
        # Token values
        val1 = seq1[i] if i < n else placeholder
        val2 = seq2[j] if j < m else placeholder
        aligned1.append(val1)
        aligned2.append(val2)

        # Token global positions
        idx1 = indexes1[i] if i < n else placeholder
        idx2 = indexes2[j] if j < m else placeholder
        aligned_idx1.append(idx1)
        aligned_idx2.append(idx2)

    similarity_score = 1 - cost[row_ind, col_ind].mean()

    return aligned1, aligned2, aligned_idx1, aligned_idx2, similarity_score

def highlight_text(paragraph,highlight_indexes,missing_map,marker_start='[[', marker_end=']]'):
    words = paragraph.replace(" %","%").split()

    highlighted_words = []

    for i, word in enumerate(words):
        if i in highlight_indexes:
            if not missing_map[i]:
                highlighted_words.append(f"{marker_start}{word}{marker_end}")
            else:
                highlighted_words.append(f"{marker_start}MISSING VALUE{marker_end}{word}")

        else:
            highlighted_words.append(word)

    highlighted_text = " ".join(highlighted_words)
    return highlighted_text


# Helper function to highlight found errors in txt file
def highlight_words(paragraph_a,paragraph_b, highlight_indexes_a,missing_map_a,highlight_indexes_b,missing_map_b,marker_start='[[', marker_end=']]', out_file='output.txt'):

    highlighted_text_a=highlight_text(paragraph_a,highlight_indexes_a,missing_map_a,marker_start,marker_end)
    highlighted_text_b=highlight_text(paragraph_b,highlight_indexes_b,missing_map_b,marker_start,marker_end)

    # Write to file
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(highlighted_text_a)
        f.write("\n\n\n")
        f.write(highlighted_text_b)


for par_num,(par_en,par_lv) in enumerate(zip(paragraphs_en,paragraphs_lv)):
    par_number=par_en["para_number"]
    number_words_en,number_words_indices_en = get_all_strings_containing_numbers(par_en["para"])
    number_words_lv,number_words_indices_lv = get_all_strings_containing_numbers(par_lv["para"])
    aligned_a, aligned_b,aligned_a_idx,aligned_b_idx, score = semantic_align(number_words_en, number_words_lv,number_words_indices_en,number_words_indices_lv)

    if aligned_a != aligned_b:
        print(f"FOUND problem in {par_num}!")
        errors_i_a=[]
        missing_map_a={}
        errors_i_b = []
        missing_map_b = {}
        for i in range(len(aligned_a)):
            if aligned_a[i]!=aligned_b[i]:
                idx_a=aligned_a_idx[i] if (aligned_a_idx[i] is not None) else aligned_b_idx[i]
                idx_b=aligned_b_idx[i] if (aligned_a_idx[i] is not None) else aligned_b_idx[i]

                missing_map_a[idx_a] = (aligned_a_idx[i] is None)
                errors_i_a.append(idx_a)
                missing_map_b[idx_b] = (aligned_b_idx[i] is None)
                errors_i_b.append(idx_b)
        if par_num==13:
            print(number_words_en)
            print(number_words_lv)
            print(aligned_a)
            print(aligned_b)

        highlight_words(par_en["para"],par_lv["para"], errors_i_a,missing_map_a,errors_i_b,missing_map_b, out_file=f'highlighted_{par_num}.txt')

