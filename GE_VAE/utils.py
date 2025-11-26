import random
import numpy as np

import torch
from torch.utils.data import Dataset


def set_seed(seed_value=42):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # Numpy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed_all(seed_value)  # Pytorch All GPU


class GeneDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_0 = self.data[idx, :]
        return x_0

    def mean_std(self):
        if torch.cuda.is_available():
            return torch.mean(self.data, dim=0), torch.std(self.data, dim=0)
        else:
            return torch.mean(self.data, dim=0), torch.std(self.data, dim=0)


def loss_function(VAELossParams, kld_weight, free_bits_nats=0.0, c=1.0):
    recon, x, mu, logvar = VAELossParams

    # 재구성항 (상수 D/2*log(2πc)는 생략 가능)
    recon_term = ((recon - x) ** 2).sum(dim=1).mean() / (2.0 * c)
    # KL(q||p), p=N(0,I)
    kld_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    loss = recon_term + kld_weight * kld_term

    return {
        "loss": loss,
        "Reconstruction_Loss": recon_term.detach(),
        "KLD": kld_term.detach(),
    }


def rowwise_pcc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a, b: [B, G]
    return: PCC per row, shape [B]
    """
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)
    return num / den


def validate(
    model, val_dataloader, device, loss_function, kld_weight: float, p_low: float = 0.10
):
    model.eval()
    total_loss_sum = 0.0
    total_count = 0
    pcc_list = []

    with torch.no_grad():
        for batch_idx, x in enumerate(val_dataloader):
            x = x.to(device, non_blocking=True)

            # GE_VAE.forward returns (recon, x, mu, logvar)
            recon, _, mu_z, logv_z = model(x)

            # 손실 집계: 배치 크기 가중 평균이 되도록
            loss_dict = loss_function((recon, x, mu_z, logv_z), kld_weight)
            batch_loss = loss_dict["loss"].item()
            B = x.size(0)
            total_loss_sum += batch_loss * B
            total_count += B

            # Per-row PCC (벡터화)
            pcc_batch = rowwise_pcc(x, recon).detach().cpu()
            pcc_list.append(pcc_batch)

    # 전체 합치기
    pcc_all = torch.cat(pcc_list) if len(pcc_list) > 0 else torch.tensor([])
    val_loss = total_loss_sum / max(1, total_count)
    pcc_row_mean = pcc_all.mean().item() if pcc_all.numel() > 0 else float("nan")
    # “pcc_low”: 하위 p_low 분위수(예: 10% 지점)
    pcc_low = (
        torch.quantile(pcc_all, p_low).item() if pcc_all.numel() > 0 else float("nan")
    )

    return {
        "val_loss": val_loss,
        "pcc_row_mean": pcc_row_mean,
    }
