import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, Tuple
from transformers import AutoModel, AutoTokenizer

import numpy as np
import pandas as pd
import os


class MoleculeModel(nn.Module):
    def __init__(self, args, context_dim=576, ge_dim=978, hidden_dim=512, timesteps=1000):
        super().__init__()
        self.n_timesteps = timesteps        # 몇 번의 diffusion step을 수행할지 (noise를 몇 번 추가할지)
        self.device = args.device
        self.mode = args.mode
        self.model_idx = args.model_idx
        
        self.latent_dim = 256
        
        self.ge_dim = ge_dim  # Gene expression dimension

        self.register_buffer("betas", torch.linspace(1e-5, 0.01, self.n_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alpha_bars_prev", torch.cat([torch.ones(1), self.alpha_bars[:-1]]))


        # posterior 계산에 필요한 계수들
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        # log variance (clip은 여기선 생략, 나중에 필요시 추가 가능)
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

        # define the molformer model
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            deterministic_eval=True
        ).to(self.device).eval()
        for param in self.molformer.parameters():
            param.requires_grad = False

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)


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

        self.denoiser_net = Denoiser(latent_dim=self.latent_dim, hidden_dim=hidden_dim, context_dim=256)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        

    def encode_context(self, basal_ge, smiles, dose, time):
        
        # Compound-specific features
        inputs = self.tokenizer(smiles, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            mol_embedding = self.molformer(**inputs).pooler_output  # (B, 768)
        mol_embedding = self.mol_ffn(mol_embedding)  # (B, 256)
        
        # Cell-specific features        
        basal_ge = self.basal_ffn(basal_ge)  # (B, 256)

        # Condition features
        dose = self.dose_ffn(dose.to(self.device))  # (B, 32)
        time = self.time_ffn(time.to(self.device))  # (B, 32)

        # Combine all features to form context vector
        context = torch.cat([mol_embedding, basal_ge, time, dose], dim=1)  # (B, 256 + 256 + 32 + 32 = 576)
        
        context = context.to(self.device)
        return self.context_encoder(context)  # (B, hidden_dim)


    def forward_posterior(self, z_0, z_t, t):
        """
        Compute q(z_{t-1} | z_t, z_0) = N(posterior_mean, posterior_variance)
        """
        # shape: (B, 1)
        coef1 = self.posterior_mean_coef1[t].view(-1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1)
        posterior_mean = coef1 * z_0 + coef2 * z_t

        posterior_log_var = self.posterior_log_variance_clipped[t].view(-1, 1)

        return posterior_mean, posterior_log_var
        
        
    def forward_diffusion(self, z_0, t):
        """
        (For mu, variance prediction model)
        Latent space z_0 to noisy latent space z_t.
        """
        noise = torch.randn_like(z_0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(self.device)
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return z_t, noise


    def reverse_diffusion(self, x, basal_ge, smiles, dose, time, t, save, count):
        
        if self.mode == "pred_mu_var":
            mu, log_var = self.denoiser(x, basal_ge, smiles, dose, time, t, save, count)
        elif self.mode == "pred_mu_v":
            mu, log_var, v, log_beta_t, log_beta_tilde = self.denoiser_pred_v(x, basal_ge, smiles, dose, time, t, save, count)
            
        sigma = torch.exp(0.5 * log_var)

        eps = torch.randn_like(mu)
        x_prev = mu + sigma * eps

        if self.mode == "pred_mu_var":
            return x_prev, mu, sigma, 
        elif self.mode == "pred_mu_v":
            return x_prev, mu, sigma, v, log_beta_t, log_beta_tilde

    
    def denoiser(self, noisy_x, basal_ge, smiles, dose, time, t, save=False, count=0):
        context = self.encode_context(basal_ge, smiles, dose, time)
        
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        out = self.denoiser_net(noisy_x, context, t_emb)  # shape: (B, latent_dim * 2)
        mu, log_var = torch.chunk(out, 2, dim=1)  # (B, latent_dim), (B, latent_dim)

        # Optional: clip log_var to prevent extreme values
        log_var = torch.clamp(log_var, min=-10, max=10)

        return mu, log_var
    

    def denoiser_pred_v(self, noisy_x, basal_ge, smiles, dose, time, t, save=False, count=0):
        context = self.encode_context(basal_ge, smiles, dose, time)
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        out = self.denoiser_net(noisy_x, context, t_emb)
        mu, v = torch.chunk(out, 2, dim=1)
        v = torch.sigmoid(v)

        beta_t = self.betas[t].view(-1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        alpha_bar_prev = self.alpha_bars[torch.clamp(t - 1, 0)].view(-1, 1)
        beta_t_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

        eps = 1e-20
        log_beta_t = torch.log(beta_t.clamp(min=eps))
        log_beta_tilde  = torch.log(beta_t_tilde.clamp(min=eps))

        log_var = v * log_beta_t + (1 - v) * log_beta_tilde

        return mu, log_var, v, log_beta_t, log_beta_tilde
    
    
    def forward_diffusion_np(self, x, t):
        """
        (For noise prediction model)
        Latent space x_0 to noisy latent space x_t.       
        """
        noise = torch.randn_like(x).to(self.device)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(self.device)
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_x, noise
    

    def reverse_diffusion_np(self, x, basal_ge, smiles, dose, time, t):
        """
        (For noise prediction model)
        Predict the noise that added to x_{t-1} using x_t.
        Return the reconstructed x_{t-1}.
        """
        predicted_noise = self.denoiser(x, basal_ge, smiles, dose, time, t)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(self.device)
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.alpha_bars[t_prev].view(-1, 1).to(self.device)

        return torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * predicted_noise
    

    def sample(self, cell, basal_ge, smiles, dose, time, batch_size, n_steps=None, save=False, count=0):
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        init_z = z.clone()  # Save initial z for potential use later
        steps = list(torch.linspace(0, self.n_timesteps - 1, n_steps or 50, dtype=torch.int64))

        for t in reversed(steps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            if self.mode == "pred_mu_var":
                z, mu, sigma = self.reverse_diffusion(z, basal_ge, smiles, dose, time, t_batch, save=save, count=count)
            elif self.mode == "pred_mu_v":
                z, mu, sigma, v, log_beta_t, log_beta_tilde = self.reverse_diffusion(z, basal_ge, smiles, dose, time, t_batch, save=save, count=count)

            if save:
                SAVE_PATH = f'./checkpoints/fold_0/model_{self.model_idx}/{count}'
                if not os.path.exists(f'{SAVE_PATH}'):
                    os.makedirs(f'{SAVE_PATH}')
                info_df = pd.DataFrame({
                    'cell': [c[0] for c in cell],
                    'smiles': smiles,
                    'dose': dose.cpu().numpy().reshape(-1),
                    'time': time.cpu().numpy().reshape(-1),
                })
                info_df.to_csv(f'{SAVE_PATH}/info.csv', index=False)
                
                torch.save(init_z, f'{SAVE_PATH}/init_z.pt')
                if (steps.index(t) % 10 == 9) or steps.index(t) == 0:
                    print(f"Step {steps.index(t)}: t={t}, x={z.mean().item()}")
                    torch.save(z, f'{SAVE_PATH}/step_{steps.index(t)}.pt')

        if self.mode == "pred_mu_var":
            return z, mu, sigma,
        elif self.mode == "pred_mu_v":
            return z, mu, sigma, v, log_beta_t, log_beta_tilde


class Denoiser(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, context_dim=256):
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
    
    def forward(self, noisy_x, context, t_emb):
        
        input_tensor = torch.cat([noisy_x, context, t_emb], dim=1)  # Concatenate inputs
        
        out = self.fc1(input_tensor)
        out = self.fc1_bn(out)
        out = self.fc1_lrelu(out)
        out = self.fc1_dropout(out)
        
        out = torch.cat([out, context], dim=1)  # 512 + 256 = 768
        out = self.fc2(out)
        out = self.fc2_bn(out)
        out = self.fc2_lrelu(out)
        out = self.fc2_dropout(out)
        
        out = torch.cat([out, context], dim=1)  # 512 + 256 = 768
        out = self.fc3(out)
        out = self.fc3_bn(out)
        out = self.fc3_lrelu(out)
        out = self.fc3_dropout(out)
        
        out = self.fc4(out)  # Final output layer
        
        return out  # Return mean and log variance

