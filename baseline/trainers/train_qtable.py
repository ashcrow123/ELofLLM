"""Van Eecke et al.: population of bidirectional dynamic Q-tables."""

from __future__ import annotations

import logging
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from baseline.config import TrainConfig
from baseline.data import Alphabet, FeatureSpace
from baseline.game import ReferentialSampler
from baseline.logging_utils import log_scalars
from baseline.models.qtable import QPopulation
from baseline.eval import evaluate_qtable, should_evaluate
from baseline.trainers.common import SuccessMeter


def train_qtable(
    cfg: TrainConfig,
    space: FeatureSpace,
    alphabet: Alphabet,
    logger: logging.Logger,
    writer: SummaryWriter,
    run_dir: str,
    test_space: FeatureSpace | None = None,
) -> None:
    logger.info(
        "qtable agents=%s concepts=%s alpha=%s",
        cfg.player_num,
        len(space.concept_names),
        cfg.alpha,
    )
    if cfg.player_num < 2 or cfg.player_num % 2 != 0:
        raise ValueError("player_num must be an even number >= 2")
    pop = QPopulation(
        n_agents=cfg.player_num,
        concept_names=space.concept_names,
        alphabet=alphabet,
        max_length=cfg.max_length,
        alpha=cfg.alpha,
        init_score=cfg.init_score,
        seed=cfg.seed,
    )
    sampler = ReferentialSampler(space.features, option_num=cfg.option_num, seed=cfg.seed)
    meter = SuccessMeter(cfg.success_window)
    pair_rng = random.Random(cfg.seed + 1)

    n_pairs = cfg.player_num // 2
    for step in tqdm(range(cfg.steps), desc="qtable"):
        ids = list(range(cfg.player_num))
        pair_rng.shuffle(ids)
        successes = []
        deltas = []
        lengths = []
        for p in range(n_pairs):
            sp, lp = ids[2 * p], ids[2 * p + 1]
            target_id, cand_ids, target_pos = sampler.sample_one()
            obj = space.concept_names[target_id]
            cand_names = [space.concept_names[int(i)] for i in cand_ids]
            word = pop.speak(pop.agents[sp], obj)
            choice = pop.listen(pop.agents[lp], word, cand_names)
            success = choice == target_pos
            mapped = pop.agents[lp].objects_for_word(word)
            known = next(iter(mapped), None) if mapped else None
            delta = pop.align(
                pop.agents[sp],
                pop.agents[lp],
                obj,
                word,
                success,
                None if success else known,
            )
            successes.append(float(success))
            deltas.append(delta)
            lengths.append(word.count("-") + 1 if word else 0)
        mean_s = sum(successes) / len(successes)
        meter.add(mean_s)
        metrics = {
            "loss/delta_q": sum(deltas) / len(deltas),
            "reward/success_rate": meter.mean(),
            "reward/success_batch": mean_s,
            "msg_len": sum(lengths) / len(lengths),
            "lexicon_size": pop.mean_lexicon_size(),
        }
        log_scalars(writer, logger, step, metrics, cfg.log_interval)
        if (step + 1) % cfg.ckpt_interval == 0 or step + 1 == cfg.steps:
            path = os.path.join(run_dir, "checkpoints", f"step_{step + 1}.pt")
            torch.save({"step": step + 1, "cfg": cfg.to_dict(), "population": pop.state_dict()}, path)
            logger.info("saved checkpoint %s", path)
        if test_space is not None and should_evaluate(cfg, step):
            evaluate_qtable(cfg, pop, space, test_space, logger, writer, run_dir, step + 1)
    writer.flush()
