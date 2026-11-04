"""Shared training helpers."""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.config import TrainConfig
from baseline.models.lstm_agents import sequence_mask


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def make_optimizer(params, cfg: TrainConfig) -> Optional[torch.optim.Optimizer]:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr)
    if cfg.optimizer == "none":
        return None
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def hinge_ranking_loss(scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    target = scores.gather(1, target_idx.unsqueeze(1))
    margin = F.relu(1.0 - target + scores)
    mask = torch.ones_like(scores)
    mask.scatter_(1, target_idx.unsqueeze(1), 0.0)
    return (margin * mask).sum(dim=1).mean()


def pointing_ce_loss(scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(scores, target_idx)


def batch_success(scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return (scores.argmax(dim=-1) == target_idx).float().mean()


def mean_message_length(token_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    _, lengths = sequence_mask(token_ids, eos_id)
    return lengths.float().mean()


class SuccessMeter:
    def __init__(self, window: int):
        self.buf: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.buf.append(float(value))

    def mean(self) -> float:
        if not self.buf:
            return 0.0
        return sum(self.buf) / len(self.buf)


def save_torch(run_dir: str, step: int, payload: Dict[str, Any]) -> str:
    path = os.path.join(run_dir, "checkpoints", f"step_{step}.pt")
    torch.save(payload, path)
    return path
