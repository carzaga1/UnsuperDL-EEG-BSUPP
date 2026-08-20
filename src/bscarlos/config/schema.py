"""VAE training configuration: dataclass + YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
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
    train_val_test_split: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    random_seed: int = 42
    annotator: str = "a1"
    checkpoint_dir: str = "models"

    def __post_init__(self) -> None:
        # Not just defensive: PyYAML's safe_load parses a bare "1e-5" (no decimal
        # point) as the *string* "1e-5", not a float. This coercion is what
        # actually makes that scientific-notation form work.
        self.n_channels = int(self.n_channels)
        self.window_size = int(self.window_size)
        self.input_dim = int(self.input_dim)
        self.latent_dim = int(self.latent_dim)
        self.beta = float(self.beta)
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.num_epochs = int(self.num_epochs)
        self.random_seed = int(self.random_seed)

        expected_input_dim = self.window_size * self.n_channels
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim ({self.input_dim}) must equal window_size * n_channels "
                f"({self.window_size} * {self.n_channels} = {expected_input_dim})"
            )


def load_training_config(path: Path | str | None = None, **overrides: Any) -> VAETrainingConfig:
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        values.update(loaded)
    values.update(overrides)
    return VAETrainingConfig(**values)
