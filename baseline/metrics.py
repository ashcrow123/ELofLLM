"""Topsim and CBM, matching statistic/statistic_methods.py on in-memory lexicons."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from baseline.data import SLOTS


def split_cv_blocks(word: str) -> List[str]:
    if not word:
        return []
    return [b for b in str(word).split("-") if b]


def edit_distance_cv_blocks(blocks1: Sequence[str], blocks2: Sequence[str]) -> int:
    m, n = len(blocks1), len(blocks2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if blocks1[i - 1] == blocks2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _slot_map(record: Mapping) -> Dict[str, List[str]]:
    return {slot: list(record.get(slot) or []) for slot in SLOTS}


def hamming_feature_matrix(records: Sequence[Mapping]) -> np.ndarray:
    """Symmetric difference of per-slot feature sets (same as get_hamming_dist_mat)."""
    feats = [_slot_map(r) for r in records]
    n = len(feats)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            a, b = feats[i], feats[j]
            if a == b:
                dist[i, j] = 0.0
            else:
                dist[i, j] = sum(len(set(a[slot]) ^ set(b[slot])) for slot in SLOTS)
    return dist


def edit_distance_matrix(words: Sequence[str]) -> np.ndarray:
    blocks = [split_cv_blocks(w) for w in words]
    n = len(blocks)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i, j] = edit_distance_cv_blocks(blocks[i], blocks[j])
    return dist


def topographic_similarity(dist_mat1: np.ndarray, dist_mat2: np.ndarray) -> float:
    assert dist_mat1.shape == dist_mat2.shape
    if dist_mat1.shape[0] < 2:
        return float("nan")
    triu_idx = np.triu_indices_from(dist_mat1, k=1)
    dist1_flat = dist_mat1[triu_idx]
    dist2_flat = dist_mat2[triu_idx]
    if dist1_flat.size == 0 or np.allclose(dist1_flat, dist1_flat[0]) or np.allclose(dist2_flat, dist2_flat[0]):
        return float("nan")
    corr, _ = spearmanr(dist1_flat, dist2_flat)
    return float(corr) if corr == corr else float("nan")


def calculate_topsim(word_dict: Mapping[str, Mapping]) -> float:
    """Spearman corr of feature Hamming vs message edit distance (calculate_topsim)."""
    records = [v for v in word_dict.values() if str(v.get("word") or "")]
    if len(records) < 3:
        return float("nan")
    hamming = hamming_feature_matrix(records)
    edits = edit_distance_matrix([str(r["word"]) for r in records])
    return topographic_similarity(hamming, edits)


def calculate_cbm(word_dict: Mapping[str, Mapping]) -> float:
    """Per-slot Hungarian CBM, matching calculate_cbm_v2 in statistic_methods.py."""
    cooccur = {attr: defaultdict(lambda: defaultdict(int)) for attr in SLOTS}
    concept_counts = {attr: defaultdict(int) for attr in SLOTS}
    symbol_set = {attr: set() for attr in SLOTS}

    for v in word_dict.values():
        symbols = set(split_cv_blocks(str(v.get("word") or "")))
        if not symbols:
            continue
        for attr in SLOTS:
            val = v.get(attr, None)
            if not isinstance(val, list) or len(val) == 0:
                continue
            concept = val[0]
            concept_counts[attr][concept] += 1
            for s in symbols:
                cooccur[attr][s][concept] += 1
                symbol_set[attr].add(s)

    attr_scores = []
    for attr in SLOTS:
        symbols = list(symbol_set[attr])
        concepts = list(concept_counts[attr].keys())
        if not symbols or not concepts:
            continue
        W = np.zeros((len(symbols), len(concepts)))
        for i, s in enumerate(symbols):
            for j, c in enumerate(concepts):
                if concept_counts[attr][c] > 0:
                    W[i, j] = cooccur[attr][s][c] / concept_counts[attr][c]
        row_ind, col_ind = linear_sum_assignment(-W)
        attr_scores.append(W[row_ind, col_ind].sum() / len(concepts))
    if not attr_scores:
        return 0.0
    return float(np.mean(attr_scores))


def mean_or_nan(values: Iterable[Optional[float]]) -> float:
    nums = [float(v) for v in values if v is not None and v == v]
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))
