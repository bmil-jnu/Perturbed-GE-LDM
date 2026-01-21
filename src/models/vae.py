"""
Gene Expression Variational Autoencoder (GE_VAE).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with linear layers."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GE_VAE(nn.Module):
    """
    Variational Autoencoder for gene expression data.
    
    Maps gene expression vectors to a latent space and back.
    Used as the compression component in the Latent Diffusion Model.
    """
    
    def __init__(self, input_dim: int = 978, latent_dim: int = 256):
        """
        Initialize GE_VAE.
        
        Args:
            input_dim: Input dimension (number of genes)
            latent_dim: Latent space dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
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

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor (B, input_dim)
            
        Returns:
            Tuple of (z, mu, logvar)
        """
        h = self.encoder(x)
        mu = self.to_mean(h)
        logvar = self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output space.
        
        Args:
            z: Latent tensor (B, latent_dim)
            
        Returns:
            Reconstructed output (B, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass.
        
        Args:
            x: Input tensor (B, input_dim)
            
        Returns:
            Tuple of (reconstruction, input, mu, logvar)
        """
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, x, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device for computation
            
        Returns:
            Sampled outputs (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
