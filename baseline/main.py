"""Train one of four referential-game baselines.

Examples:
  python baseline/main.py --method stgs
  python baseline/main.py --method reinforce --steps 10000 --batch_size 64
  python baseline/main.py --method quantize --device cuda
  python baseline/main.py --method qtable --player_num 10 --count 400
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from baseline.config import METHOD_DEFAULTS, TrainConfig, build_config
from baseline.data import Alphabet, build_split_feature_spaces, select_letters, vocab_report
from baseline.logging_utils import setup_run
from baseline.trainers import TRAINERS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BRM referential-game baselines")
    p.add_argument("--method", required=True, choices=sorted(METHOD_DEFAULTS))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--count", type=int, default=None, help="number of v2 concepts")
    p.add_argument("--option_num", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--letter_num", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--ckpt_interval", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None, help="0 disables mid-train eval; always eval at last step")
    p.add_argument("--eval_games", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None)
    p.add_argument("--embed_dim", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--tau0", type=float, default=None)
    p.add_argument("--learn_tau", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None)
    p.add_argument("--speaker_entropy", type=float, default=None)
    p.add_argument("--listener_entropy", type=float, default=None)
    p.add_argument("--vq_beta", type=float, default=None)
    p.add_argument("--player_num", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--init_score", type=float, default=None)
    p.add_argument("--success_window", type=int, default=None)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg: TrainConfig = build_config(args)
    seed_all(cfg.seed)

    space, test_space = build_split_feature_spaces(seed=cfg.seed, count=cfg.count)
    letter_list = select_letters(num=cfg.letter_num, seed=cfg.seed)
    alphabet = Alphabet(letter_list)

    logger, writer, run_dir = setup_run(cfg)
    logger.info("feat_dim=%s train=%s test=%s slot_sizes=%s", space.feat_dim, len(space.concept_names), len(test_space.concept_names), vocab_report(space))
    ones = space.features.sum(axis=1)
    logger.info("ones_per_object min/mean/max=%.0f/%.2f/%.0f (expect ~10)", ones.min(), ones.mean(), ones.max())
    logger.info("letters=%s", letter_list)

    TRAINERS[cfg.method](cfg, space, alphabet, logger, writer, run_dir, test_space=test_space)
    writer.close()
    logger.info("training finished")


if __name__ == "__main__":
    main()
