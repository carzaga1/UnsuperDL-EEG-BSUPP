from pathlib import Path
from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from bscarlos.architectures.datasets import EEGDataset
from bscarlos.architectures.vae import VAE
from bscarlos.config.schema import VAETrainingConfig, load_training_config


def train_vae(
    windows: np.ndarray | torch.Tensor,
    config: VAETrainingConfig | None = None,
    checkpoint_path: Path | str | None = None,
) -> dict[str, list[float]]:
    """Trains a VAE model on windowed EEG data and saves a checkpoint."""
    if config is None:
        config = load_training_config()

    if checkpoint_path is None:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"vae_{config.channel_set}.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Determine compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(windows, torch.Tensor):
        windows = windows.cpu().numpy()

    # Train/Val/Test split if sufficient samples exist
    if len(windows) >= 10:
        val_test_prop = config.train_val_test_split[1] + config.train_val_test_split[2]
        x_train, _ = train_test_split(windows, test_size=val_test_prop, random_state=config.random_seed)
    else:
        x_train = windows

    train_dataset = EEGDataset(x_train)
    batch_size = min(config.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE(input_dim=config.input_dim, latent_dim=config.latent_dim, output_dim=config.input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    recon_losses: list[float] = []
    kl_losses: list[float] = []
    total_losses: list[float] = []

    model.train()
    for epoch in range(config.num_epochs):
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_total_loss = 0.0

        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar, _ = model(data)

            recon_loss = torch.sum(torch.abs(recon_batch - data))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + kl_loss * config.beta

            total_loss.backward()
            optimizer.step()

            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_total_loss += total_loss.item()

        n_samples = len(train_dataset)
        recon_losses.append(train_recon_loss / n_samples)
        kl_losses.append(train_kl_loss / n_samples)
        total_losses.append(train_total_loss / n_samples)

    checkpoint_payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "input_dim": config.input_dim,
        "latent_dim": config.latent_dim,
        "config": config.__dict__,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    return {
        "recon_loss": recon_losses,
        "kl_loss": kl_losses,
        "total_loss": total_losses,
    }
