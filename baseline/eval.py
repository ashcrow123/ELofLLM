"""Frozen evaluation: train success, generalization rate, topsim, CBM."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Any, Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

from baseline.config import TrainConfig
from baseline.data import Alphabet, FeatureSpace
from baseline.game import ReferentialSampler
from baseline.metrics import calculate_cbm, calculate_topsim, mean_or_nan
from baseline.models.lstm_agents import LSTMListener
from baseline.models.quantize import QuantizeSpeaker
from baseline.models.qtable import BidirectionalQTable, QPopulation
from baseline.trainers.common import resolve_device


def log_eval(
    logger: logging.Logger,
    writer: SummaryWriter,
    step: int,
    metrics: Dict[str, Any],
) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        fv = float(value)
        if math.isfinite(fv):
            writer.add_scalar(key, fv, step)
    parts = [f"{k}={float(v):.4f}" for k, v in metrics.items() if v is not None]
    logger.info("eval step %s | %s", step, " ".join(parts))


def greedy_message(speaker: torch.nn.Module, feats: torch.Tensor, max_length: int):
    """Return (token_ids, channel_embeddings). Quantize keeps the VQ codebook channel."""
    if isinstance(speaker, QuantizeSpeaker):
        token_ids, z_seq, _ = speaker.generate(feats, max_length)
        return token_ids, z_seq
    if hasattr(speaker, "generate_greedy"):
        return speaker.generate_greedy(feats, max_length), None
    token_ids, *_ = speaker.generate(feats, max_length)
    return token_ids, None


@torch.no_grad()
def collect_neural_lexicon(
    speaker: torch.nn.Module,
    space: FeatureSpace,
    alphabet: Alphabet,
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, object]]:
    speaker.eval()
    n = len(space.concept_names)
    if n == 0:
        return {}
    words: List[str] = []
    feats = torch.from_numpy(space.features).to(device)
    for start in range(0, n, batch_size):
        chunk = feats[start : start + batch_size]
        token_ids, _ = greedy_message(speaker, chunk, max_length)
        for row in token_ids:
            words.append(alphabet.decode(row.tolist()))
    return space.to_word_dict(words)


@torch.no_grad()
def neural_success_rate(
    speaker: torch.nn.Module,
    listener: LSTMListener,
    space: FeatureSpace,
    option_num: int,
    max_length: int,
    n_games: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    if space.features.shape[0] < option_num or n_games <= 0:
        return float("nan")
    speaker.eval()
    listener.eval()
    sampler = ReferentialSampler(space.features, option_num=option_num, seed=seed)
    hits = 0.0
    seen = 0
    remaining = n_games
    while remaining > 0:
        bsz = min(batch_size, remaining)
        batch = sampler.sample(bsz, device)
        token_ids, channel_emb = greedy_message(speaker, batch.target_feats, max_length)
        scores = listener.scores(
            token_ids, batch.candidate_feats, embeddings=channel_emb
        )
        hits += float((scores.argmax(dim=-1) == batch.target_idx).float().sum().item())
        seen += bsz
        remaining -= bsz
    return hits / seen if seen else float("nan")


def collect_qtable_lexicon(agent: BidirectionalQTable, space: FeatureSpace) -> Dict[str, Dict[str, object]]:
    words = []
    keep_idx = []
    for i, name in enumerate(space.concept_names):
        word = agent.best_word(name)
        if not word:
            continue
        words.append(word)
        keep_idx.append(i)
    if not keep_idx:
        return {}
    subset_names = [space.concept_names[i] for i in keep_idx]
    subset_slots = [space.slot_lists[i] for i in keep_idx]
    tmp = FeatureSpace(
        slot_vocabs=space.slot_vocabs,
        slot_index=space.slot_index,
        offsets=space.offsets,
        feat_dim=space.feat_dim,
        concept_names=subset_names,
        features=space.features[keep_idx],
        name_to_idx={n: i for i, n in enumerate(subset_names)},
        slot_lists=subset_slots,
    )
    return tmp.to_word_dict(words)


def qtable_success_rate(
    pop: QPopulation,
    space: FeatureSpace,
    option_num: int,
    n_games: int,
    seed: int,
) -> float:
    if space.features.shape[0] < option_num or n_games <= 0 or len(pop.agents) < 2:
        return float("nan")
    sampler = ReferentialSampler(space.features, option_num=option_num, seed=seed)
    pair_rng = random.Random(seed + 17)
    hits = 0
    for i in range(n_games):
        sp, lp = pair_rng.sample(range(len(pop.agents)), 2)
        target_id, cand_ids, target_pos = sampler.sample_one()
        obj = space.concept_names[target_id]
        cand_names = [space.concept_names[int(j)] for j in cand_ids]
        word = pop.agents[sp].best_word(obj) or ""
        choice = pop.listen(pop.agents[lp], word, cand_names)
        hits += int(choice == target_pos)
    return hits / n_games


def linguistic_metrics(word_dict: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    return {
        "eval/topsim": calculate_topsim(word_dict),
        "eval/cbm": calculate_cbm(word_dict),
        "eval/lexicon_named": float(len(word_dict)),
    }


def dump_eval(
    run_dir: str,
    step: int,
    metrics: Dict[str, Any],
    lexicon: Dict[str, Dict[str, object]],
) -> None:
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    payload = {"step": step, "metrics": metrics}
    with open(os.path.join(eval_dir, f"step_{step}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(eval_dir, f"lexicon_step_{step}.json"), "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=2, ensure_ascii=False)


def evaluate_neural(
    cfg: TrainConfig,
    speaker: torch.nn.Module,
    listener: LSTMListener,
    train_space: FeatureSpace,
    test_space: FeatureSpace,
    alphabet: Alphabet,
    logger: logging.Logger,
    writer: SummaryWriter,
    run_dir: str,
    step: int,
) -> Dict[str, float]:
    device = resolve_device(cfg.device)
    speaker.eval()
    listener.eval()
    batch_size = max(1, min(cfg.batch_size, cfg.eval_batch_size))
    lexicon = collect_neural_lexicon(
        speaker, train_space, alphabet, cfg.max_length, device, batch_size
    )
    metrics = linguistic_metrics(lexicon)
    metrics["eval/train_success"] = neural_success_rate(
        speaker,
        listener,
        train_space,
        cfg.option_num,
        cfg.max_length,
        cfg.eval_games,
        batch_size,
        cfg.seed + 101,
        device,
    )
    metrics["eval/gen_rate"] = neural_success_rate(
        speaker,
        listener,
        test_space,
        cfg.option_num,
        cfg.max_length,
        cfg.eval_games,
        batch_size,
        cfg.seed + 202,
        device,
    )
    dump_eval(run_dir, step, metrics, lexicon)
    log_eval(logger, writer, step, metrics)
    speaker.train()
    listener.train()
    return metrics


def evaluate_qtable(
    cfg: TrainConfig,
    pop: QPopulation,
    train_space: FeatureSpace,
    test_space: FeatureSpace,
    logger: logging.Logger,
    writer: SummaryWriter,
    run_dir: str,
    step: int,
) -> Dict[str, float]:
    topsims = []
    cbms = []
    named = []
    lexica = {}
    for i, agent in enumerate(pop.agents):
        lexicon = collect_qtable_lexicon(agent, train_space)
        lexica[str(i)] = lexicon
        ling = linguistic_metrics(lexicon)
        topsims.append(ling["eval/topsim"])
        cbms.append(ling["eval/cbm"])
        named.append(ling["eval/lexicon_named"])
    metrics = {
        "eval/topsim": mean_or_nan(topsims),
        "eval/cbm": mean_or_nan(cbms),
        "eval/lexicon_named": mean_or_nan(named),
        "eval/train_success": qtable_success_rate(
            pop, train_space, cfg.option_num, cfg.eval_games, cfg.seed + 101
        ),
        "eval/gen_rate": qtable_success_rate(
            pop, test_space, cfg.option_num, cfg.eval_games, cfg.seed + 202
        ),
    }
    dump_eval(run_dir, step, metrics, lexica)
    log_eval(logger, writer, step, metrics)
    return metrics


def should_evaluate(cfg: TrainConfig, step: int) -> bool:
    last = step + 1 == cfg.steps
    if last:
        return True
    return bool(cfg.eval_interval) and (step + 1) % cfg.eval_interval == 0
