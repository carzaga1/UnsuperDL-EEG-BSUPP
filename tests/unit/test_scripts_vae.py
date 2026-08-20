from pathlib import Path
import numpy as np
import pandas as pd

from scripts.predict_vae import main as predict_main
from scripts.train_vae import main as train_main


def test_scripts_train_and_predict_cli(tmp_path: Path):
    checkpoint_path = tmp_path / "models" / "cli_vae.pt"
    data_path = tmp_path / "data.npy"
    out_preds_path = tmp_path / "preds.csv"

    # Save a dummy windows array
    windows = np.random.rand(20, 3072).astype(np.float32)
    np.save(data_path, windows)

    # 1. Test train_vae CLI
    train_args = [
        "--config",
        "config/vae_6ch.yml",
        "--data-path",
        str(data_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--num-epochs",
        "1",
        "--batch-size",
        "8",
    ]
    train_main(train_args)
    assert checkpoint_path.exists()

    # 2. Test predict_vae CLI
    predict_args = [
        "--config",
        "config/vae_6ch.yml",
        "--checkpoint-path",
        str(checkpoint_path),
        "--data-path",
        str(data_path),
        "--output-path",
        str(out_preds_path),
    ]
    predict_main(predict_args)
    assert out_preds_path.exists()

    df_preds = pd.read_csv(out_preds_path)
    assert "reconstruction_error" in df_preds.columns
    assert "predicted_label" in df_preds.columns
    assert len(df_preds) == 20
