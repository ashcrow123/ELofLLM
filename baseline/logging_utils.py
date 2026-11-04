"""File logging plus a TensorBoard writer."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from baseline.config import TrainConfig


def setup_run(cfg: TrainConfig) -> tuple[logging.Logger, SummaryWriter, str]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = cfg.run_name or f"{cfg.method}_{stamp}"
    run_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    logger = logging.getLogger(f"baseline.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    logger.info("Run directory: %s", run_dir)
    logger.info("Config dumped to %s", cfg_path)

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    return logger, writer, run_dir


def log_scalars(
    writer: SummaryWriter,
    logger: logging.Logger,
    step: int,
    metrics: Dict[str, Any],
    log_interval: int,
) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        writer.add_scalar(key, float(value), step)
    if step == 0 or (step + 1) % log_interval == 0:
        parts = [f"{k}={float(v):.4f}" for k, v in metrics.items() if v is not None]
        logger.info("step %s | %s", step + 1, " ".join(parts))
