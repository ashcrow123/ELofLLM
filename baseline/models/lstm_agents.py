"""Shared LSTM speaker and listener for discrete sequential messages."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax_sample(logits: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    u = torch.rand_like(logits).clamp(min=1e-6, max=1 - 1e-6)
    gumbel = -torch.log(-torch.log(u))
    return torch.softmax((logits + gumbel) / tau, dim=-1)


def straight_through_gumbel(logits: torch.Tensor, tau: torch.Tensor | float):
    y_soft = gumbel_softmax_sample(logits, tau)
    index = y_soft.argmax(dim=-1)
    y_hard = F.one_hot(index, num_classes=logits.size(-1)).to(dtype=y_soft.dtype)
    y = y_hard + y_soft - y_soft.detach()
    return y, index, y_soft


def sequence_mask(token_ids: torch.Tensor, eos_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep tokens up to and including the first EOS. Returns keep-mask (B,T) and lengths (B,)."""
    bsz, tmax = token_ids.shape
    is_eos = token_ids == eos_id
    first_eos = torch.where(
        is_eos.any(dim=1),
        is_eos.float().argmax(dim=1),
        torch.full((bsz,), tmax - 1, device=token_ids.device, dtype=torch.long),
    )
    steps = torch.arange(tmax, device=token_ids.device).unsqueeze(0)
    keep = steps <= first_eos.unsqueeze(1)
    lengths = first_eos + 1
    return keep, lengths


def last_hidden_from_packed(
    lstm: nn.LSTM,
    embeddings: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    lengths_cpu = lengths.clamp(min=1).cpu()
    packed = nn.utils.rnn.pack_padded_sequence(
        embeddings, lengths_cpu, batch_first=True, enforce_sorted=False
    )
    packed_out, _ = lstm(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(
        packed_out, batch_first=True, total_length=embeddings.size(1)
    )
    idx = (lengths.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
    return out.gather(1, idx).squeeze(1)


class LSTMSpeaker(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        eos_id: int,
        learn_tau: bool = False,
        tau0: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.feat_to_h = nn.Linear(feat_dim, hidden_size)
        self.feat_to_c = nn.Linear(feat_dim, hidden_size)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.start = nn.Parameter(torch.zeros(embed_dim))
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.learn_tau = learn_tau
        self.tau0 = tau0
        self.tau_head = nn.Linear(hidden_size, 1)

    def _tau(self, hidden: torch.Tensor, tau: float) -> torch.Tensor:
        if not self.learn_tau:
            return torch.full((hidden.size(0), 1), tau, device=hidden.device)
        inv = F.softplus(self.tau_head(hidden)) + self.tau0
        return 1.0 / inv.clamp(min=1e-3)

    def _init_state(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.feat_to_h(feats)).unsqueeze(0)
        c = torch.tanh(self.feat_to_c(feats)).unsqueeze(0)
        return h, c

    def generate_stgs(self, feats: torch.Tensor, max_length: int, tau: float):
        bsz = feats.size(0)
        h, c = self._init_state(feats)
        x = self.start.unsqueeze(0).expand(bsz, -1).unsqueeze(1)
        tokens = []
        onehots = []
        entropies = []
        finished = torch.zeros(bsz, dtype=torch.bool, device=feats.device)
        for _ in range(max_length):
            out, (h, c) = self.lstm(x, (h, c))
            hid = out.squeeze(1)
            logits = self.proj(hid)
            step_tau = self._tau(hid, tau)
            y, index, y_soft = straight_through_gumbel(logits, step_tau)
            logp = torch.log(y_soft.clamp(min=1e-9))
            entropies.append(-(y_soft * logp).sum(dim=-1))
            index = torch.where(finished, torch.full_like(index, self.eos_id), index)
            y = F.one_hot(index, self.vocab_size).float() + y - y.detach()
            finished = finished | (index == self.eos_id)
            tokens.append(index)
            onehots.append(y)
            x = y @ self.embed.weight
            x = x.unsqueeze(1)
        token_ids = torch.stack(tokens, dim=1)
        return token_ids, torch.stack(onehots, dim=1), torch.stack(entropies, dim=1)

    def generate_reinforce(self, feats: torch.Tensor, max_length: int):
        bsz = feats.size(0)
        h, c = self._init_state(feats)
        x = self.start.unsqueeze(0).expand(bsz, -1).unsqueeze(1)
        tokens = []
        logps = []
        entropies = []
        finished = torch.zeros(bsz, dtype=torch.bool, device=feats.device)
        for _ in range(max_length):
            out, (h, c) = self.lstm(x, (h, c))
            hid = out.squeeze(1)
            logits = self.proj(hid)
            dist = torch.distributions.Categorical(logits=logits)
            index = dist.sample()
            logp = dist.log_prob(index)
            ent = dist.entropy()
            index = torch.where(finished, torch.full_like(index, self.eos_id), index)
            logp = torch.where(finished, torch.zeros_like(logp), logp)
            ent = torch.where(finished, torch.zeros_like(ent), ent)
            finished = finished | (index == self.eos_id)
            tokens.append(index)
            logps.append(logp)
            entropies.append(ent)
            x = self.embed(index).unsqueeze(1)
        token_ids = torch.stack(tokens, dim=1)
        return token_ids, torch.stack(logps, dim=1), torch.stack(entropies, dim=1)

    def generate_greedy(self, feats: torch.Tensor, max_length: int) -> torch.Tensor:
        bsz = feats.size(0)
        h, c = self._init_state(feats)
        x = self.start.unsqueeze(0).expand(bsz, -1).unsqueeze(1)
        tokens = []
        finished = torch.zeros(bsz, dtype=torch.bool, device=feats.device)
        for _ in range(max_length):
            out, (h, c) = self.lstm(x, (h, c))
            index = self.proj(out.squeeze(1)).argmax(dim=-1)
            index = torch.where(finished, torch.full_like(index, self.eos_id), index)
            finished = finished | (index == self.eos_id)
            tokens.append(index)
            x = self.embed(index).unsqueeze(1)
        return torch.stack(tokens, dim=1)


class LSTMListener(nn.Module):
    def __init__(self, feat_dim: int, vocab_size: int, embed_dim: int, hidden_size: int, eos_id: int):
        super().__init__()
        self.eos_id = eos_id
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.msg_out = nn.Linear(hidden_size, hidden_size)
        self.feat_enc = nn.Linear(feat_dim, hidden_size)

    def scores(
        self,
        token_ids: torch.Tensor,
        candidate_feats: torch.Tensor,
        token_onehot: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        keep, lengths = sequence_mask(token_ids, self.eos_id)
        if embeddings is not None:
            emb = embeddings
        elif token_onehot is not None:
            emb = token_onehot @ self.embed.weight
        else:
            emb = self.embed(token_ids)
        emb = emb * keep.unsqueeze(-1)
        last = last_hidden_from_packed(self.lstm, emb, lengths)
        g = self.msg_out(last)
        cand = self.feat_enc(candidate_feats)
        return (cand * g.unsqueeze(1)).sum(dim=-1)

    def pointing_policy(self, token_ids: torch.Tensor, candidate_feats: torch.Tensor):
        logits = self.scores(token_ids, candidate_feats)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), logits
