"""VAE training configuration: dataclass + YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VAETrainingConfig:
    channel_set: str = "6ch"
    n_channels: int = 6
    window_size: int = 512
    input_dim: int = 3072
    latent_dim: int = 32
    beta: float = 0.8
    learning_rate: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 100
    train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    random_seed: int = 42
    annotator: str = "a1"
    checkpoint_dir: str = "vae_checkpoints"

    def __post_init__(self) -> None:
        expected_input_dim = self.window_size * self.n_channels
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim ({self.input_dim}) must equal window_size * n_channels "
                f"({self.window_size} * {self.n_channels} = {expected_input_dim})"
            )


def load_training_config(path: Path | None = None, **overrides: Any) -> VAETrainingConfig:
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        if "train_val_test_split" in loaded:
            loaded["train_val_test_split"] = tuple(loaded["train_val_test_split"])
        values.update(loaded)
    values.update(overrides)
    return VAETrainingConfig(**values)
