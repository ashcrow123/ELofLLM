"""Public defaults plus per-method paper defaults.

Paper sources:
- stgs: Havrylov & Titov, NeurIPS 2017
- reinforce: Lazaridou et al., ICLR 2018 appendix B
- quantize: Carmeli et al., AAAI 2023 (quantized channel; batch/opt follow stgs scale)
- qtable: Van Eecke et al., Journal of Language Evolution 2022/23
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


COMMON_DEFAULTS: Dict[str, Any] = dict(
    seed=40,
    count=400,
    option_num=5,
    max_length=3,
    letter_num=10,
    steps=20000,
    device="cuda",
    run_name=None,
    log_dir="baseline_runs",
    log_interval=50,
    ckpt_interval=1000,
    success_window=100,
    eval_interval=1000,
    eval_games=256,
    eval_batch_size=64,
)

METHOD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "stgs": dict(
        batch_size=128,
        lr=1e-3,
        optimizer="adam",
        embed_dim=256,
        hidden_size=512,
        tau=1.2,
        tau0=0.2,
        learn_tau=True,
    ),
    "reinforce": dict(
        batch_size=32,
        lr=1e-4,
        optimizer="rmsprop",
        embed_dim=256,
        hidden_size=512,
        speaker_entropy=0.01,
        listener_entropy=0.001,
        baseline_momentum=0.99,
    ),
    "quantize": dict(
        batch_size=128,
        lr=1e-3,
        optimizer="adam",
        embed_dim=256,
        hidden_size=512,
        vq_beta=0.25,
    ),
    "qtable": dict(
        batch_size=1,
        player_num=10,
        alpha=0.5,
        init_score=0.5,
        lr=0.5,
        optimizer="none",
        embed_dim=0,
        hidden_size=0,
    ),
}


@dataclass
class TrainConfig:
    method: str
    seed: int = 40
    count: int = 400
    option_num: int = 5
    max_length: int = 3
    letter_num: int = 10
    steps: int = 20000
    batch_size: int = 128
    device: str = "cuda"
    run_name: Optional[str] = None
    log_dir: str = "baseline_runs"
    log_interval: int = 50
    ckpt_interval: int = 1000
    success_window: int = 100
    eval_interval: int = 1000
    eval_games: int = 256
    eval_batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"
    embed_dim: int = 256
    hidden_size: int = 512
    tau: float = 1.2
    tau0: float = 0.2
    learn_tau: bool = True
    speaker_entropy: float = 0.01
    listener_entropy: float = 0.001
    baseline_momentum: float = 0.99
    vq_beta: float = 0.25
    player_num: int = 10
    alpha: float = 0.5
    init_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_config(args: Namespace) -> TrainConfig:
    method = args.method
    if method not in METHOD_DEFAULTS:
        raise ValueError(f"Unknown method {method}. Choose from {list(METHOD_DEFAULTS)}")
    values: Dict[str, Any] = {}
    values.update(COMMON_DEFAULTS)
    values.update(METHOD_DEFAULTS[method])
    values["method"] = method
    for f in fields(TrainConfig):
        if f.name == "method":
            continue
        if hasattr(args, f.name):
            raw = getattr(args, f.name)
            if raw is not None:
                values[f.name] = raw
    allowed = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in values.items() if k in allowed})
