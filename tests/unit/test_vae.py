import torch

from bscarlos.architectures import VAE, Decoder, Encoder, reparameterize


def test_vae_architecture():
    batch_size = 4
    input_dim = 3072
    latent_dim = 32
    output_dim = 3072

    encoder = Encoder(input_dim, latent_dim)
    decoder = Decoder(latent_dim, output_dim)
    vae = VAE(input_dim, latent_dim, output_dim)

    x = torch.randn(batch_size, input_dim)

    # Test Encoder
    mu, logvar = encoder(x)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

    # Test reparameterize
    z = reparameterize(mu, logvar)
    assert z.shape == (batch_size, latent_dim)

    # Test Decoder
    recon = decoder(z)
    assert recon.shape == (batch_size, output_dim)
    # Output of decoder should be in [0, 1] due to sigmoid
    assert (recon >= 0.0).all() and (recon <= 1.0).all()

    # Test VAE forward pass
    recon_x, mu, logvar, z = vae(x)
    assert recon_x.shape == (batch_size, output_dim)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)

    # Test VAE loss function
    loss = vae.loss_function(recon_x, x, mu, logvar, beta=0.1)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
