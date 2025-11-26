import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return x + self.block(x)


class GE_VAE(nn.Module):
    def __init__(self, input_dim=978, latent_dim=256):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),  # 안정적 regularization
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU()
        )

        self.to_mean = nn.Linear(256, latent_dim)
        self.to_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),

            ResidualBlock(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),

            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 512),
            nn.LeakyReLU(),

            nn.Linear(512, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.to_mean(h)
        logvar = self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, x, mu, logvar
