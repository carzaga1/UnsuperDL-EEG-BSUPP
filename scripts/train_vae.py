#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from bscarlos.architectures.datasets import normalize_eeg_data, reshape_to_windows
from bscarlos.config import load_training_config
from bscarlos.training import train_vae


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE on EEG windows.")
    parser.add_argument("--config", type=str, default="config/vae_6ch.yml", help="Path to YAML config file")
    parser.add_argument("--data-path", type=str, default=None, help="Path to parquet or npy data file")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Output checkpoint path")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent_dim")
    parser.add_argument("--beta", type=float, default=None, help="Override beta")
    parser.add_argument("--channel-set", type=str, default=None, help="Override channel_set")
    return parser.parse_args(args)


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    overrides = {}
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.latent_dim is not None:
        overrides["latent_dim"] = args.latent_dim
    if args.beta is not None:
        overrides["beta"] = args.beta
    if args.channel_set is not None:
        overrides["channel_set"] = args.channel_set

    config = load_training_config(args.config, **overrides)

    if args.data_path and Path(args.data_path).exists():
        data_path = Path(args.data_path)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
            if "ground_truth" in df.columns:
                df = df.drop(columns=["ground_truth"])
            raw_data = normalize_eeg_data(df.values)
            windows = reshape_to_windows(raw_data, window_size=config.window_size, n_channels=config.n_channels)
        elif data_path.suffix in [".npy", ".npz"]:
            windows = np.load(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
    else:
        np.random.seed(config.random_seed)
        windows = np.random.rand(100, config.input_dim).astype(np.float32)

    history = train_vae(windows, config=config, checkpoint_path=args.checkpoint_path)
    print(f"Training completed successfully. Final total loss: {history['total_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
