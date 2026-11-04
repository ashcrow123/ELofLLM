"""Vector-quantized recurrent speaker (Carmeli et al.)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Embedding(n_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = z.reshape(-1, self.dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices).view_as(z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + self.beta * commit_loss
        z_st = z + (z_q - z).detach()
        return z_st, indices.view(*z.shape[:-1]), vq_loss


class QuantizeSpeaker(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        eos_id: int,
        vq_beta: float = 0.25,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.feat_to_h = nn.Linear(feat_dim, hidden_size)
        self.feat_to_c = nn.Linear(feat_dim, hidden_size)
        self.start = nn.Parameter(torch.zeros(embed_dim))
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.to_z = nn.Linear(hidden_size, embed_dim)
        self.vq = VectorQuantizer(vocab_size, embed_dim, beta=vq_beta)

    def generate(self, feats: torch.Tensor, max_length: int):
        bsz = feats.size(0)
        h = torch.tanh(self.feat_to_h(feats)).unsqueeze(0)
        c = torch.tanh(self.feat_to_c(feats)).unsqueeze(0)
        x = self.start.unsqueeze(0).expand(bsz, -1).unsqueeze(1)
        tokens = []
        quantized = []
        vq_losses = []
        finished = torch.zeros(bsz, dtype=torch.bool, device=feats.device)
        for _ in range(max_length):
            out, (h, c) = self.lstm(x, (h, c))
            z = self.to_z(out.squeeze(1))
            z_q, index, vq_loss = self.vq(z)
            index = torch.where(finished, torch.full_like(index, self.eos_id), index)
            finished = finished | (index == self.eos_id)
            tokens.append(index)
            quantized.append(z_q)
            vq_losses.append(vq_loss)
            x = z_q.unsqueeze(1)
        token_ids = torch.stack(tokens, dim=1)
        z_seq = torch.stack(quantized, dim=1)
        vq_loss = torch.stack(vq_losses).mean()
        return token_ids, z_seq, vq_loss

    def generate_greedy(self, feats: torch.Tensor, max_length: int) -> torch.Tensor:
        token_ids, _, _ = self.generate(feats, max_length)
        return token_ids
