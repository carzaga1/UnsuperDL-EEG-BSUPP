import math
from pathlib import Path
import numpy as np

from bscarlos.architectures import VAE
from bscarlos.config import VAETrainingConfig
from bscarlos.inference import load_model, predict_labels, reconstruction_error
from bscarlos.training import train_vae


def test_predict_vae_pipeline(tmp_path: Path):
    num_windows = 15
    input_dim = 3072
    windows = np.random.rand(num_windows, input_dim).astype(np.float32)

    checkpoint_path = tmp_path / "models" / "vae_test.pt"
    config = VAETrainingConfig(
        num_epochs=1,
        batch_size=8,
        checkpoint_dir=str(tmp_path / "models"),
    )

    train_vae(windows, config=config, checkpoint_path=checkpoint_path)

    # 1. Test load_model
    model = load_model(checkpoint_path, config=config)
    assert isinstance(model, VAE)

    # 2. Test reconstruction_error
    errors = reconstruction_error(model, windows)
    assert isinstance(errors, np.ndarray)
    assert errors.shape == (num_windows,)
    assert (errors >= 0.0).all()
    for err in errors:
        assert not math.isnan(err)
        assert not math.isinf(err)

    # 3. Test predict_labels
    labels = predict_labels(errors, threshold=None)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (num_windows,)
    assert set(np.unique(labels)).issubset({0, 1})

    # Fixed threshold test
    fixed_threshold = float(np.mean(errors))
    labels_fixed = predict_labels(errors, threshold=fixed_threshold)
    assert labels_fixed.shape == (num_windows,)
