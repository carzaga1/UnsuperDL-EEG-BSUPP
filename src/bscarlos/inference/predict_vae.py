from pathlib import Path
import numpy as np
import torch

from bscarlos.architectures.vae import VAE
from bscarlos.config.schema import VAETrainingConfig


def load_model(checkpoint_path: Path | str, config: VAETrainingConfig | None = None) -> VAE:
    """Loads a trained VAE model from a checkpoint file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_dim = checkpoint.get("input_dim")
        latent_dim = checkpoint.get("latent_dim")
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        input_dim = None
        latent_dim = None

    if config is not None:
        input_dim = config.input_dim
        latent_dim = config.latent_dim

    if input_dim is None or latent_dim is None:
        input_dim = 3072
        latent_dim = 32

    model = VAE(input_dim=input_dim, latent_dim=latent_dim, output_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def reconstruction_error(model: VAE, windows: np.ndarray | torch.Tensor) -> np.ndarray:
    """Computes L1 reconstruction error per window."""
    device = next(model.parameters()).device
    model.eval()

    if isinstance(windows, np.ndarray):
        x = torch.tensor(windows, dtype=torch.float32)
    else:
        x = windows.to(dtype=torch.float32)

    x = x.to(device)

    with torch.no_grad():
        recon_x, _, _, _ = model(x)
        errors = torch.sum(torch.abs(recon_x - x), dim=1)

    return errors.cpu().numpy()


def predict_labels(errors: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Predicts binary suppression/burst labels from reconstruction errors."""
    if threshold is None:
        threshold = float(np.percentile(errors, 75))

    return (errors > threshold).astype(int)
