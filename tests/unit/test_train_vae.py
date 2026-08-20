import math
from pathlib import Path
import numpy as np

from bscarlos.config import VAETrainingConfig
from bscarlos.training import train_vae


def test_train_vae_smoke(tmp_path: Path):
    num_windows = 20
    input_dim = 3072
    windows = np.random.rand(num_windows, input_dim).astype(np.float32)

    checkpoint_path = tmp_path / "models" / "vae_test.pt"

    config = VAETrainingConfig(
        num_epochs=2,
        batch_size=8,
        checkpoint_dir=str(tmp_path / "models"),
    )

    history = train_vae(windows, config=config, checkpoint_path=checkpoint_path)

    assert "recon_loss" in history
    assert "kl_loss" in history
    assert "total_loss" in history
    assert len(history["total_loss"]) == 2

    for loss_list in history.values():
        for val in loss_list:
            assert isinstance(val, float)
            assert not math.isnan(val)
            assert not math.isinf(val)

    assert checkpoint_path.exists()
