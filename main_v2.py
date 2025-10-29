import json
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
import Levenshtein

# ----------------------
# JSON Loading
# ----------------------
def read_paragraphs_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data[0]["para"]

paragraphs_en = read_paragraphs_from_json("eval_sample_en.json")
paragraphs_lv = read_paragraphs_from_json("eval_sample_lv.json")

# ----------------------
# Text cleaning / numeric extraction
# ----------------------
def clean_par(paragraph):
    # paragraph = re.sub(r'(?<=\d)\s+(?=\d)', '', paragraph)
    words = paragraph.replace(" %", "%").replace("  %", "%").split()
    return words

def get_all_strings_containing_numbers(paragraph):
    words = clean_par(paragraph)
    number_words = []
    indexes = []
    for i, word in enumerate(words):
        tokens = re.findall(r'\b\d+%|\b\w+\b', word)
        for token in tokens:
            if re.search(r'\d', token):
                number_words.append(token)
                indexes.append(i)
    return number_words, indexes

def clean_token(token):
    return token

def numeric_similarity(a, b):
    if not a or not b:
        return 0.0
    a_norm, b_norm = clean_token(a), clean_token(b)
    if a_norm == b_norm:
        return 1.0
    a_nums, b_nums = set(a_norm.split()), set(b_norm.split())
    if not a_nums or not b_nums:
        return 0.0
    inter = len(a_nums & b_nums)
    union = len(a_nums | b_nums)
    return inter / union

def semantic_align(seq1, seq2, indexes1, indexes2, placeholder=None):
    n, m = len(seq1), len(seq2)
    size = max(n, m)
    cost = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i < n and j < m:
                sim = numeric_similarity(seq1[i], seq2[j])
                cost[i, j] = 1 - sim
            else:
                cost[i, j] = 1
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned1, aligned2 = [], []
    aligned_idx1, aligned_idx2 = [], []
    for i, j in zip(row_ind, col_ind):
        val1 = seq1[i] if i < n else placeholder
        val2 = seq2[j] if j < m else placeholder
        aligned1.append(val1)
        aligned2.append(val2)
        idx1 = indexes1[i] if i < n else placeholder
        idx2 = indexes2[j] if j < m else placeholder
        aligned_idx1.append(idx1)
        aligned_idx2.append(idx2)
    similarity_score = 1 - cost[row_ind, col_ind].mean()
    return aligned1, aligned2, aligned_idx1, aligned_idx2, similarity_score

# ----------------------
# Highlighting functions
# ----------------------
def highlight_text(paragraph, highlight_indexes, missing_map, marker_start='[[', marker_end=']]'):
    words = clean_par(paragraph)
    highlighted_words = []
    for i, word in enumerate(words):
        if i % 10 == 0:
            highlighted_words.append("\n")
        if i in highlight_indexes:
            if not missing_map.get(i, False):
                highlighted_words.append(f"{marker_start}{word}{marker_end}")
            else:
                highlighted_words.append(f"{marker_start}MISSING VALUE{marker_end}{word}")
        else:
            highlighted_words.append(word)
    return " ".join(highlighted_words)

def highlight_words(paragraph_a, paragraph_b, highlight_indexes_a, missing_map_a, highlight_indexes_b, missing_map_b, marker_start='[[', marker_end=']]', out_file='output.txt'):
    highlighted_text_a = highlight_text(paragraph_a, highlight_indexes_a, missing_map_a, marker_start, marker_end)
    highlighted_text_b = highlight_text(paragraph_b, highlight_indexes_b, missing_map_b, marker_start, marker_end)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(highlighted_text_a)
        f.write("\n\n\n")
        f.write(highlighted_text_b)

# ----------------------
# Levenshtein-based error detection
# ----------------------
def filter_out_words(input_text: str):
    cleaned = input_text.replace('(', '').replace(')', '')
    split = cleaned.split(' ')
    no_num = [w for w in split if not w.isdigit()]
    short_words = [w for w in no_num if 1 < len(w) < 15 and w.isupper()]
    return short_words

def levenstein_distance(paragraph_one, paragraph_two):
    words_one = filter_out_words(paragraph_one)
    words_two = filter_out_words(paragraph_two)
    return [
        (w1, w2)
        for w1 in words_one
        for w2 in words_two
        if Levenshtein.distance(w1, w2) == 1 and len(w1) == len(w2)
    ]

# ----------------------
# Main loop
# ----------------------
for par_num, (par_en, par_lv) in enumerate(zip(paragraphs_en, paragraphs_lv)):
    # Numeric alignment
    number_words_en, number_words_indices_en = get_all_strings_containing_numbers(par_en["para"])
    number_words_lv, number_words_indices_lv = get_all_strings_containing_numbers(par_lv["para"])
    aligned_a, aligned_b, aligned_a_idx, aligned_b_idx, score = semantic_align(
        number_words_en, number_words_lv, number_words_indices_en, number_words_indices_lv
    )

    errors_i_a, missing_map_a = [], {}
    errors_i_b, missing_map_b = [], {}

    # Numeric mismatches
    for i in range(len(aligned_a)):
        if aligned_a[i] != aligned_b[i]:
            idx_a = aligned_a_idx[i] if aligned_a_idx[i] is not None else aligned_b_idx[i]
            idx_b = aligned_b_idx[i] if aligned_b_idx[i] is not None else aligned_a_idx[i]
            missing_map_a[idx_a] = (aligned_a_idx[i] is None)
            missing_map_b[idx_b] = (aligned_b_idx[i] is None)
            errors_i_a.append(idx_a)
            errors_i_b.append(idx_b)

    # Levenshtein mismatches
    leven_pairs = levenstein_distance(par_en["para"], par_lv["para"])
    words_en = clean_par(par_en["para"])
    words_lv = clean_par(par_lv["para"])
    for w1, w2 in leven_pairs:
        # find first occurrence positions in the paragraph
        if w1 in words_en:
            idx = words_en.index(w1)
            errors_i_a.append(idx)
            missing_map_a[idx] = False
        if w2 in words_lv:
            idx = words_lv.index(w2)
            errors_i_b.append(idx)
            missing_map_b[idx] = False

    if errors_i_a or errors_i_b:
        highlight_words(
            par_en["para"], par_lv["para"],
            errors_i_a, missing_map_a,
            errors_i_b, missing_map_b,
            out_file=f'highlighted_{par_num}.txt'
        )
