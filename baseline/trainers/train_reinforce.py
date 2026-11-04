"""Lazaridou et al.: LSTM + REINFORCE on speaker and listener pointing."""

from __future__ import annotations

import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from baseline.config import TrainConfig
from baseline.data import Alphabet, FeatureSpace
from baseline.game import ReferentialSampler
from baseline.logging_utils import log_scalars
from baseline.models.lstm_agents import LSTMListener, LSTMSpeaker
from baseline.eval import evaluate_neural, should_evaluate
from baseline.trainers.common import (
    SuccessMeter,
    batch_success,
    make_optimizer,
    mean_message_length,
    resolve_device,
    save_torch,
)


def train_reinforce(
    cfg: TrainConfig,
    space: FeatureSpace,
    alphabet: Alphabet,
    logger: logging.Logger,
    writer: SummaryWriter,
    run_dir: str,
    test_space: FeatureSpace | None = None,
) -> None:
    device = resolve_device(cfg.device)
    logger.info("reinforce device=%s feat_dim=%s vocab=%s", device, space.feat_dim, alphabet.vocab_size)
    speaker = LSTMSpeaker(
        space.feat_dim,
        alphabet.vocab_size,
        cfg.embed_dim,
        cfg.hidden_size,
        alphabet.eos_id,
    ).to(device)
    listener = LSTMListener(
        space.feat_dim,
        alphabet.vocab_size,
        cfg.embed_dim,
        cfg.hidden_size,
        alphabet.eos_id,
    ).to(device)
    opt = make_optimizer(list(speaker.parameters()) + list(listener.parameters()), cfg)
    sampler = ReferentialSampler(space.features, option_num=cfg.option_num, seed=cfg.seed)
    meter = SuccessMeter(cfg.success_window)
    baseline = 0.0

    speaker.train()
    listener.train()
    for step in tqdm(range(cfg.steps), desc="reinforce"):
        batch = sampler.sample(cfg.batch_size, device)
        token_ids, logp_s, ent_s = speaker.generate_reinforce(batch.target_feats, cfg.max_length)
        action, logp_l, ent_l, logits = listener.pointing_policy(token_ids, batch.candidate_feats)
        reward = (action == batch.target_idx).float()
        baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * reward.mean().item()
        advantage = reward - baseline
        pg = -(advantage.detach() * (logp_s.sum(dim=1) + logp_l)).mean()
        entropy = -(cfg.speaker_entropy * ent_s.mean() + cfg.listener_entropy * ent_l.mean())
        loss = pg + entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(speaker.parameters()) + list(listener.parameters()), 5.0)
        opt.step()

        with torch.no_grad():
            greedy = batch_success(logits, batch.target_idx).item()
            meter.add(reward.mean().item())
            metrics = {
                "loss/total": loss.item(),
                "loss/policy": pg.item(),
                "reward/success_rate": meter.mean(),
                "reward/success_batch": reward.mean().item(),
                "reward/greedy_success": greedy,
                "reward/baseline": baseline,
                "msg_len": mean_message_length(token_ids, alphabet.eos_id).item(),
                "entropy/speaker": ent_s.mean().item(),
                "entropy/listener": ent_l.mean().item(),
                "lr": cfg.lr,
            }
        log_scalars(writer, logger, step, metrics, cfg.log_interval)
        if (step + 1) % cfg.ckpt_interval == 0 or step + 1 == cfg.steps:
            path = save_torch(
                run_dir,
                step + 1,
                {
                    "step": step + 1,
                    "cfg": cfg.to_dict(),
                    "speaker": speaker.state_dict(),
                    "listener": listener.state_dict(),
                    "opt": opt.state_dict(),
                    "baseline": baseline,
                },
            )
            logger.info("saved checkpoint %s", path)
        if test_space is not None and should_evaluate(cfg, step):
            evaluate_neural(
                cfg, speaker, listener, space, test_space, alphabet, logger, writer, run_dir, step + 1
            )
    writer.flush()
