"""BRM v2 feature space: full per-slot vocab, v2 only chooses which index is 1."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SLOTS: Tuple[str, ...] = (
    "encyclopaedic",
    "function",
    "smell",
    "sound",
    "tactile",
    "taste",
    "taxonomic",
    "visual_colour",
    "visual_form_and_surface",
    "visual_motion",
)
EMPTY_TOKEN = "<EMPTY>"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONCS_FEATS_PATH = PROJECT_ROOT / "data" / "McRae-BRM-InPress" / "CONCS_FEATS_concstats_brm.xlsx"


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Feature"] = df["Feature"].str.replace("_", " ", regex=False)
    df["BR_Label"] = df["BR_Label"].str.replace("-", "_", regex=False)
    df["Feature"] = df["Feature"].str.replace(r"^beh - ", "living behavior: ", regex=True)
    df["Feature"] = df["Feature"].str.replace(r"^inbeh - ", "non-living behavior: ", regex=True)
    return df


def load_concs_feats() -> pd.DataFrame:
    df = pd.read_excel(CONCS_FEATS_PATH)
    return _normalize_frame(df)


def select_letters(num: int = 10, seed: Optional[int] = None) -> List[str]:
    rng = random.Random(seed)
    all_letters = [c + v for c in "bcdfghjklmnpqrstvwxyz" for v in "aeiou"]
    return rng.sample(all_letters, k=num)


def build_slot_vocabs(df: pd.DataFrame) -> Dict[str, List[str]]:
    """All feature strings that appear in each slot, plus an empty token."""
    vocabs: Dict[str, List[str]] = {}
    for slot in SLOTS:
        values = sorted(
            {
                str(v)
                for v in df.loc[df["BR_Label"] == slot, "Feature"].dropna().unique()
            }
        )
        vocabs[slot] = [EMPTY_TOKEN] + values
    return vocabs


def load_v2_pairs(
    df: pd.DataFrame,
    seed: Optional[int] = None,
    count: Optional[int] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Each concept keeps only the highest Prod_Freq feature per slot."""
    pairs: Dict[str, Dict[str, List[str]]] = {}
    for concept, group in df.groupby("Concept"):
        slots = {slot: [] for slot in SLOTS}
        for category, category_group in group.groupby("BR_Label"):
            if category not in slots or category_group.empty:
                continue
            max_row = category_group.loc[category_group["Prod_Freq"].idxmax()]
            slots[category].append(str(max_row["Feature"]))
        pairs[str(concept)] = slots
    if count is not None:
        rng = random.Random(seed)
        keys = rng.sample(list(pairs.keys()), k=min(count, len(pairs)))
        pairs = {k: pairs[k] for k in keys}
    return pairs


def encode_slots(
    slots: Dict[str, List[str]],
    slot_index: Dict[str, Dict[str, int]],
    offsets: Dict[str, int],
    feat_dim: int,
) -> np.ndarray:
    vec = np.zeros(feat_dim, dtype=np.float32)
    for slot in SLOTS:
        chosen = slots.get(slot) or []
        token = chosen[0] if chosen else EMPTY_TOKEN
        local = slot_index[slot]
        if token not in local:
            token = EMPTY_TOKEN
        vec[offsets[slot] + local[token]] = 1.0
    return vec


@dataclass
class FeatureSpace:
    slot_vocabs: Dict[str, List[str]]
    slot_index: Dict[str, Dict[str, int]]
    offsets: Dict[str, int]
    feat_dim: int
    concept_names: List[str]
    features: np.ndarray  # (N, feat_dim) float32
    name_to_idx: Dict[str, int]
    slot_lists: List[Dict[str, List[str]]]

    def encode_one(self, slots: Dict[str, List[str]]) -> np.ndarray:
        return encode_slots(slots, self.slot_index, self.offsets, self.feat_dim)

    def to_word_dict(self, words: Sequence[str]) -> Dict[str, Dict[str, object]]:
        if len(words) != len(self.concept_names):
            raise ValueError("Need one word per concept")
        records: Dict[str, Dict[str, object]] = {}
        for name, slots, word in zip(self.concept_names, self.slot_lists, words):
            rec: Dict[str, object] = {"word": word}
            for slot in SLOTS:
                rec[slot] = list(slots.get(slot) or [])
            records[name] = rec
        return records


def _vocab_layout(df: pd.DataFrame):
    slot_vocabs = build_slot_vocabs(df)
    slot_index = {
        slot: {tok: i for i, tok in enumerate(vocab)}
        for slot, vocab in slot_vocabs.items()
    }
    offsets: Dict[str, int] = {}
    cursor = 0
    for slot in SLOTS:
        offsets[slot] = cursor
        cursor += len(slot_vocabs[slot])
    return slot_vocabs, slot_index, offsets, cursor


def _space_from_pairs(
    pairs: Dict[str, Dict[str, List[str]]],
    slot_vocabs: Dict[str, List[str]],
    slot_index: Dict[str, Dict[str, int]],
    offsets: Dict[str, int],
    feat_dim: int,
) -> FeatureSpace:
    names = sorted(pairs.keys())
    slot_lists = [pairs[name] for name in names]
    if names:
        features = np.stack(
            [encode_slots(pairs[name], slot_index, offsets, feat_dim) for name in names],
            axis=0,
        )
    else:
        features = np.zeros((0, feat_dim), dtype=np.float32)
    return FeatureSpace(
        slot_vocabs=slot_vocabs,
        slot_index=slot_index,
        offsets=offsets,
        feat_dim=feat_dim,
        concept_names=names,
        features=features,
        name_to_idx={n: i for i, n in enumerate(names)},
        slot_lists=slot_lists,
    )


def build_split_feature_spaces(
    seed: Optional[int] = 40,
    count: Optional[int] = 400,
) -> Tuple[FeatureSpace, FeatureSpace]:
    """Train objects = `count` sample; test = remaining McRae concepts.

    Matches `get_splited_data` in data_loader/BRM_loader.py. Slot vocabs still
    come from the full table, so v2 only chooses which index is 1.
    """
    df = load_concs_feats()
    slot_vocabs, slot_index, offsets, feat_dim = _vocab_layout(df)
    all_pairs = load_v2_pairs(df, seed=None, count=None)
    names = list(all_pairs.keys())
    if count is None or count >= len(names):
        train_pairs = all_pairs
        test_pairs: Dict[str, Dict[str, List[str]]] = {}
    else:
        rng = random.Random(seed)
        train_keys = rng.sample(names, k=count)
        train_set = set(train_keys)
        train_pairs = {k: all_pairs[k] for k in train_keys}
        test_pairs = {k: all_pairs[k] for k in names if k not in train_set}
    train = _space_from_pairs(train_pairs, slot_vocabs, slot_index, offsets, feat_dim)
    test = _space_from_pairs(test_pairs, slot_vocabs, slot_index, offsets, feat_dim)
    return train, test


def build_feature_space(
    seed: Optional[int] = 40,
    count: Optional[int] = 400,
) -> FeatureSpace:
    train, _ = build_split_feature_spaces(seed=seed, count=count)
    return train


def vocab_report(space: FeatureSpace) -> Dict[str, int]:
    return {slot: len(space.slot_vocabs[slot]) for slot in SLOTS}


class Alphabet:
    def __init__(self, letter_list: Sequence[str]):
        self.letters = list(letter_list)
        self.eos = "<EOS>"
        self.tokens = self.letters + [self.eos]
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.eos_id = self.stoi[self.eos]
        self.n_letters = len(self.letters)
        self.vocab_size = len(self.tokens)

    def decode(self, ids: Sequence[int]) -> str:
        pieces = []
        for i in ids:
            if int(i) == self.eos_id:
                break
            pieces.append(self.tokens[int(i)])
        return "-".join(pieces)

    def random_word(self, max_length: int, rng: random.Random) -> str:
        length = rng.randint(1, max_length)
        return "-".join(rng.choice(self.letters) for _ in range(length))
