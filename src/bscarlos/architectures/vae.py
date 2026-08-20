import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network for VAE."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mu = nn.Linear(256, latent_dim)
        self.fc3_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        logvar = self.fc3_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), -1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        reconstruction = torch.sigmoid(self.fc3(z))
        return reconstruction


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick to sample from N(mu, var)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    """Variational Autoencoder (VAE) for EEG signal reconstruction."""

    def __init__(self, input_dim: int, latent_dim: int, output_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar, z

    def loss_function(
        self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.1
    ) -> torch.Tensor:
        """Computes total VAE loss (L1 reconstruction loss + beta * KL divergence)."""
        recon_loss = torch.sum(torch.abs(recon_x - x))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss
