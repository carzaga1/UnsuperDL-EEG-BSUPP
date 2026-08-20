#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from bscarlos.architectures.datasets import reshape_to_windows
from bscarlos.config import load_training_config
from bscarlos.inference import load_model, predict_labels, reconstruction_error


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAE inference on EEG windows.")
    parser.add_argument("--config", type=str, default="config/vae_6ch.yml", help="Path to YAML config file")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--data-path", type=str, default=None, help="Path to input parquet or npy file")
    parser.add_argument("--threshold", type=float, default=None, help="Suppression threshold")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save output predictions (.npy or .csv)")
    return parser.parse_args(args)


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    config = load_training_config(args.config)

    model = load_model(args.checkpoint_path, config=config)

    if args.data_path and Path(args.data_path).exists():
        data_path = Path(args.data_path)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
            if "ground_truth" in df.columns:
                df = df.drop(columns=["ground_truth"])
            raw_data = df.values
            windows = reshape_to_windows(raw_data, window_size=config.window_size, n_channels=config.n_channels)
        elif data_path.suffix in [".npy", ".npz"]:
            windows = np.load(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
    else:
        np.random.seed(config.random_seed)
        windows = np.random.rand(50, config.input_dim).astype(np.float32)

    errors = reconstruction_error(model, windows)
    labels = predict_labels(errors, threshold=args.threshold)

    print(f"Inference completed for {len(labels)} windows. Positive (suppression) ratio: {labels.mean():.2%}")

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            df_out = pd.DataFrame({"reconstruction_error": errors, "predicted_label": labels})
            df_out.to_csv(out_path, index=False)
        else:
            np.save(out_path, labels)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
