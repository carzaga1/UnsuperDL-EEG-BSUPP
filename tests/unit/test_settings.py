import pytest

from bscarlos.config import VAETrainingConfig, load_training_config


def test_load_training_config_defaults():
    config = load_training_config()

    assert config.window_size == 512
    assert config.latent_dim == 32
    assert config.beta == 0.8
    assert config.learning_rate == 1e-5
    assert config.batch_size == 64
    assert config.num_epochs == 100
    assert config.input_dim == config.window_size * config.n_channels


def test_mismatched_input_dim_raises():
    with pytest.raises(ValueError):
        VAETrainingConfig(window_size=512, n_channels=6, input_dim=999)
