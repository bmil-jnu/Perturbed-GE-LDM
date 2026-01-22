"""
Diffusion model for gene expression prediction.
"""

import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """
    Cosine schedule from Improved DDPM (Nichol & Dhariwal, 2021).
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta from being too small near t=0
        max_beta: Maximum beta value to prevent signal destruction
    
    Returns:
        Tensor of shape (timesteps,) containing beta values
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize so alpha_bar_0 = 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, max_beta)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-5, beta_end: float = 0.01) -> torch.Tensor:
    """
    Linear schedule (original DDPM).
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Tensor of shape (timesteps,) containing beta values
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model for predicting perturbed gene expression.
    
    This model performs diffusion in the latent space of a VAE and conditions
    on molecular structure, basal gene expression, dose, and time.
    """
    
    def __init__(
        self,
        args,
        context_dim: int = 576,
        ge_dim: int = 978,
        hidden_dim: int = 512,
        timesteps: int = 1000
    ):
        """
        Initialize LatentDiffusionModel.
        
        Args:
            args: Arguments object with model configuration
            context_dim: Dimension of context embedding
            ge_dim: Gene expression dimension
            hidden_dim: Hidden layer dimension
            timesteps: Number of diffusion timesteps
        """
        super().__init__()
        self.n_timesteps = timesteps
        self.device = args.device
        self.mode = args.mode
        self.model_idx = getattr(args, 'model_idx', 0)
        
        self.latent_dim = 256
        self.ge_dim = ge_dim
        
        # Beta scheduling
        beta_schedule = getattr(args, 'beta_schedule', 'cosine')
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.n_timesteps, max_beta=0.01)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(self.n_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alpha_bars_prev", torch.cat([torch.ones(1), self.alpha_bars[:-1]]))

        # Posterior coefficients
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)
        )

        # MolFormer for molecular embeddings
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            deterministic_eval=True
        ).to(self.device).eval()
        for param in self.molformer.parameters():
            param.requires_grad = False

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
        )

        # Feature encoders
        self.mol_ffn = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
        )
        
        self.basal_ffn = nn.Sequential(
            nn.Linear(978, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
        )

        self.dose_ffn = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        
        self.time_ffn = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
        )

        self.denoiser_net = Denoiser(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            context_dim=256
        )

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

    def encode_context(
        self,
        basal_ge: torch.Tensor,
        smiles: List[str],
        dose: torch.Tensor,
        time: torch.Tensor,
        mol_emb_cached: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode conditioning information into context vector.
        
        Args:
            basal_ge: Basal gene expression (B, 978)
            smiles: List of SMILES strings
            dose: Dose values (B, 1)
            time: Time values (B, 1)
            mol_emb_cached: Optional precomputed molecular embeddings (B, 768)
            
        Returns:
            Context embedding (B, 256)
        """
        # Compound embedding
        if mol_emb_cached is not None:
            mol_embedding = mol_emb_cached.to(self.device)
        else:
            inputs = self.tokenizer(smiles, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                mol_embedding = self.molformer(**inputs).pooler_output
        
        mol_embedding = self.mol_ffn(mol_embedding)  # (B, 256)
        
        # Cell features
        basal_ge = self.basal_ffn(basal_ge)  # (B, 256)

        # Condition features
        dose = self.dose_ffn(dose.to(self.device))  # (B, 32)
        time = self.time_ffn(time.to(self.device))  # (B, 32)

        # Combine features
        context = torch.cat([mol_embedding, basal_ge, time, dose], dim=1)  # (B, 576)
        context = context.to(self.device)
        
        return self.context_encoder(context)  # (B, 256)

    def forward_posterior(
        self,
        z_0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute q(z_{t-1} | z_t, z_0) = N(posterior_mean, posterior_variance).
        
        Args:
            z_0: Clean latent (B, latent_dim)
            z_t: Noisy latent (B, latent_dim)
            t: Timesteps (B,)
            
        Returns:
            Tuple of (posterior_mean, posterior_log_variance)
        """
        coef1 = self.posterior_mean_coef1[t].view(-1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1)
        posterior_mean = coef1 * z_0 + coef2 * z_t
        posterior_log_var = self.posterior_log_variance_clipped[t].view(-1, 1)
        return posterior_mean, posterior_log_var

    def forward_diffusion(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to z_0 to get z_t.
        
        Args:
            z_0: Clean latent (B, latent_dim)
            t: Timesteps (B,)
            
        Returns:
            Tuple of (noisy_latent, noise)
        """
        noise = torch.randn_like(z_0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(self.device)
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return z_t, noise

    def denoiser(
        self,
        noisy_x: torch.Tensor,
        basal_ge: torch.Tensor,
        smiles: List[str],
        dose: torch.Tensor,
        time: torch.Tensor,
        t: torch.Tensor,
        save: bool = False,
        count: int = 0,
        mol_emb_cached: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and log variance for reverse diffusion.
        
        Args:
            noisy_x: Noisy latent (B, latent_dim)
            basal_ge: Basal gene expression (B, 978)
            smiles: List of SMILES strings
            dose: Dose values (B, 1)
            time: Time values (B, 1)
            t: Timesteps (B,)
            save: Whether to save intermediate results
            count: Counter for saving
            mol_emb_cached: Optional precomputed embeddings
            
        Returns:
            Tuple of (predicted_mean, predicted_log_variance)
        """
        context = self.encode_context(basal_ge, smiles, dose, time, mol_emb_cached)
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        out = self.denoiser_net(noisy_x, context, t_emb)
        mu, log_var = torch.chunk(out, 2, dim=1)
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        return mu, log_var

    def denoiser_pred_v(
        self,
        noisy_x: torch.Tensor,
        basal_ge: torch.Tensor,
        smiles: List[str],
        dose: torch.Tensor,
        time: torch.Tensor,
        t: torch.Tensor,
        beta_t: Optional[torch.Tensor] = None,
        tilde_beta_t: Optional[torch.Tensor] = None,
        save: bool = False,
        count: int = 0,
        mol_emb_cached: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict mean and variance using v-parameterization.
        
        Returns:
            Tuple of (mu, log_var, v, log_beta_t, log_beta_tilde)
        """
        context = self.encode_context(basal_ge, smiles, dose, time, mol_emb_cached)
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        out = self.denoiser_net(noisy_x, context, t_emb)
        mu, v = torch.chunk(out, 2, dim=1)
        v = torch.sigmoid(v)

        if beta_t is None or tilde_beta_t is None:
            beta_t = self.betas[t].view(-1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1)
            alpha_bar_prev = self.alpha_bars[torch.clamp(t - 1, 0)].view(-1, 1)
            tilde_beta_t = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

        eps = 1e-20
        log_beta_t = torch.log(beta_t.clamp(min=eps))
        log_beta_tilde = torch.log(tilde_beta_t.clamp(min=eps))
        log_var = v * log_beta_t + (1 - v) * log_beta_tilde

        return mu, log_var, v, log_beta_t, log_beta_tilde

    def reverse_diffusion(
        self,
        x: torch.Tensor,
        basal_ge: torch.Tensor,
        smiles: List[str],
        dose: torch.Tensor,
        time: torch.Tensor,
        t: torch.Tensor,
        beta_t: Optional[torch.Tensor] = None,
        tilde_beta_t: Optional[torch.Tensor] = None,
        save: bool = False,
        count: int = 0,
        mol_emb_cached: Optional[torch.Tensor] = None
    ):
        """
        Single step of reverse diffusion.
        """
        if self.mode == "pred_mu_var":
            mu, log_var = self.denoiser(
                x, basal_ge, smiles, dose, time, t, save, count, mol_emb_cached
            )
        elif self.mode == "pred_mu_v":
            mu, log_var, v, log_beta_t, log_beta_tilde = self.denoiser_pred_v(
                x, basal_ge, smiles, dose, time, t, beta_t, tilde_beta_t, save, count, mol_emb_cached
            )
            
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        x_prev = mu + sigma * eps

        if self.mode == "pred_mu_var":
            return x_prev, mu, sigma
        elif self.mode == "pred_mu_v":
            return x_prev, mu, sigma, v, log_beta_t, log_beta_tilde

    def sample(
        self,
        cell: List[str],
        basal_ge: torch.Tensor,
        smiles: List[str],
        dose: torch.Tensor,
        time: torch.Tensor,
        batch_size: int,
        n_steps: Optional[int] = None,
        save: bool = False,
        count: int = 0,
        mol_emb_cached: Optional[torch.Tensor] = None
    ):
        """
        Sample from the diffusion model.
        
        Args:
            cell: Cell IDs
            basal_ge: Basal gene expression
            smiles: SMILES strings
            dose: Dose values
            time: Time values
            batch_size: Number of samples
            n_steps: Number of sampling steps
            save: Whether to save intermediate steps
            count: Counter for saving
            mol_emb_cached: Precomputed embeddings
            
        Returns:
            Sampled latents and intermediate values
        """
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        init_z = z.clone()
        steps = list(torch.linspace(0, self.n_timesteps - 1, n_steps or 50, dtype=torch.int64))

        for t in reversed(steps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            if self.mode == "pred_mu_var":
                z, mu, sigma = self.reverse_diffusion(
                    z, basal_ge, smiles, dose, time, t_batch, save=save, count=count,
                    mol_emb_cached=mol_emb_cached
                )
            elif self.mode == "pred_mu_v":
                z, mu, sigma, v, log_beta_t, log_beta_tilde = self.reverse_diffusion(
                    z, basal_ge, smiles, dose, time, t_batch, save=save, count=count,
                    mol_emb_cached=mol_emb_cached
                )

            if save:
                self._save_intermediate(
                    z, init_z, cell, smiles, dose, time, steps.index(t), t, count
                )

        if self.mode == "pred_mu_var":
            return z, mu, sigma
        elif self.mode == "pred_mu_v":
            return z, mu, sigma, v, log_beta_t, log_beta_tilde

    def _save_intermediate(
        self,
        z: torch.Tensor,
        init_z: torch.Tensor,
        cell: List[str],
        smiles: List[str],
        dose: torch.Tensor,
        time: torch.Tensor,
        step_idx: int,
        t: int,
        count: int
    ) -> None:
        """Save intermediate sampling results."""
        SAVE_PATH = f'./checkpoints/fold_0/model_{self.model_idx}/{count}'
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        info_df = pd.DataFrame({
            'cell': [c[0] if isinstance(c, (list, tuple)) else c for c in cell],
            'smiles': smiles,
            'dose': dose.cpu().numpy().reshape(-1),
            'time': time.cpu().numpy().reshape(-1),
        })
        info_df.to_csv(f'{SAVE_PATH}/info.csv', index=False)
        
        torch.save(init_z, f'{SAVE_PATH}/init_z.pt')
        if (step_idx % 10 == 9) or step_idx == 0:
            print(f"Step {step_idx}: t={t}, x={z.mean().item()}")
            torch.save(z, f'{SAVE_PATH}/step_{step_idx}.pt')


class Denoiser(nn.Module):
    """
    Denoiser network for predicting clean latent from noisy input.
    """
    
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, context_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(latent_dim + context_dim + 128, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.fc1_lrelu = nn.LeakyReLU()
        self.fc1_dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim + context_dim, hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)
        self.fc2_lrelu = nn.LeakyReLU()
        self.fc2_dropout = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim + context_dim, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim)
        self.fc3_lrelu = nn.LeakyReLU()
        self.fc3_dropout = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_dim, latent_dim * 2)
    
    def forward(
        self,
        noisy_x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            noisy_x: Noisy latent (B, latent_dim)
            context: Context embedding (B, context_dim)
            t_emb: Time embedding (B, 128)
            
        Returns:
            Output tensor (B, latent_dim * 2) containing mean and log_var
        """
        input_tensor = torch.cat([noisy_x, context, t_emb], dim=1)
        
        out = self.fc1(input_tensor)
        out = self.fc1_bn(out)
        out = self.fc1_lrelu(out)
        out = self.fc1_dropout(out)
        
        out = torch.cat([out, context], dim=1)
        out = self.fc2(out)
        out = self.fc2_bn(out)
        out = self.fc2_lrelu(out)
        out = self.fc2_dropout(out)
        
        out = torch.cat([out, context], dim=1)
        out = self.fc3(out)
        out = self.fc3_bn(out)
        out = self.fc3_lrelu(out)
        out = self.fc3_dropout(out)
        
        out = self.fc4(out)
        
        return out
