"""Sample referential-game batches from a FeatureSpace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ReferentialBatch:
    target_feats: torch.Tensor  # (B, D)
    candidate_feats: torch.Tensor  # (B, K, D)
    target_idx: torch.Tensor  # (B,) index of target within K
    target_ids: torch.Tensor  # (B,) concept row in the dataset
    candidate_ids: torch.Tensor  # (B, K)


class ReferentialSampler:
    def __init__(self, features: np.ndarray, option_num: int = 5, seed: int = 0):
        if option_num < 2:
            raise ValueError("option_num must be >= 2")
        self.features = features.astype(np.float32)
        self.n = features.shape[0]
        self.k = option_num
        if self.n < self.k:
            raise ValueError(f"Need at least {self.k} concepts, got {self.n}")
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, device: torch.device) -> ReferentialBatch:
        targets = self.rng.integers(0, self.n, size=batch_size)
        candidate_ids = np.empty((batch_size, self.k), dtype=np.int64)
        target_pos = np.empty(batch_size, dtype=np.int64)
        all_idx = np.arange(self.n)
        for i, t in enumerate(targets):
            mask = all_idx != int(t)
            others = self.rng.choice(all_idx[mask], size=self.k - 1, replace=False)
            pos = int(self.rng.integers(0, self.k))
            row = np.empty(self.k, dtype=np.int64)
            row[:pos] = others[:pos]
            row[pos] = t
            row[pos + 1 :] = others[pos:]
            candidate_ids[i] = row
            target_pos[i] = pos
        cand_feats = self.features[candidate_ids]
        tgt_feats = self.features[targets]
        return ReferentialBatch(
            target_feats=torch.from_numpy(tgt_feats).to(device),
            candidate_feats=torch.from_numpy(cand_feats).to(device),
            target_idx=torch.from_numpy(target_pos).to(device),
            target_ids=torch.from_numpy(targets.astype(np.int64)).to(device),
            candidate_ids=torch.from_numpy(candidate_ids).to(device),
        )

    def sample_one(self) -> tuple[int, np.ndarray, int]:
        """Single game for tabular agents: target id, candidate ids, target position."""
        batch = self.sample(1, torch.device("cpu"))
        return (
            int(batch.target_ids[0].item()),
            batch.candidate_ids[0].cpu().numpy(),
            int(batch.target_idx[0].item()),
        )
