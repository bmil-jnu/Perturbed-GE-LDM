# Perturbed-GE-LDM

**Predicting Condition-Aware Drug-Induced Transcriptional Responses via a Latent Diffusion Model**

**Chaewon Kim, Sunyong Yoo**

---

## Overview

This repository contains the official implementation of a **Latent Diffusion Model (LDM)** for predicting drug-induced gene expression changes. The model takes basal gene expression, molecular structure (SMILES), dose, and treatment time as conditions to predict perturbed transcriptional responses.

<p align="center">
  <img src="./figure/model_architecture.jpg" width="80%" alt="Model Architecture">
</p>

## Key Features

- **Conditional Latent Diffusion**: Generates perturbed gene expression in VAE latent space
- **Multi-condition Support**: Incorporates cell type, drug structure, dose, and time
- **MolFormer Integration**: Uses pretrained molecular embeddings for drug representation
- **Distributed Training**: Supports multi-GPU training with PyTorch DDP

---

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 12.0

### Setup

```bash
git clone https://github.com/your-username/Perturbed-GE-LDM.git
cd Perturbed-GE-LDM

# Create conda environment
conda env create -f environment.yaml
conda activate ldm_lincs

# Or install via pip
pip install -r requirements.txt
```

---

## Dataset and Checkpoints

### LINCS L1000

The original L1000 dataset for drug-induced transcriptional profiles is available from Gene Expression Omnibus: [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)

In this study, we further processed the data provided by [PRnet](https://github.com/Perturbation-Response-Prediction/PRnet).

📥 **Download the processed data**: [Google Drive](https://drive.google.com/file/d/1Su6RZWbxS961ZpFxH4qU9tjth2IPxXsO/view?usp=sharing)

### Pretrained Model

📥 **Download the pretrained LDM checkpoint**: [Google Drive](https://drive.google.com/file/d/1MBQi7eMlwe7ZICKxLw6aiF1G8KQQlFv6/view?usp=sharing)

Place the downloaded file at `checkpoints/ldm/best_ldm.pt`.

### File Structure

Place the preprocessed data in the root directory, and the checkpoint in the `checkpoints/ldm/` directory:

```
Perturbed-GE-LDM/
├── Lincs_L1000.h5ad
└── checkpoints/
    └── ldm/
        └── best_ldm.pt  # Download from Google Drive
```

---

## Usage

### Training

```bash
# Basic training
python main.py train --config configs/base.yaml

# With custom parameters
python main.py train --epochs 200 --batch_size 512 --lr 1e-3

# Distributed training (multi-GPU)
python main.py train --parallel --world_size 4
```

### Evaluation

```bash
# Evaluate trained model
python main.py eval --checkpoint checkpoints/ldm/best_ldm.pt
```

### Prediction

```bash
# Generate predictions
python main.py predict --checkpoint checkpoints/ldm/best_ldm.pt --output predictions.csv
```

---

## Project Structure

```
Perturbed-GE-LDM/
├── main.py                 # Single entry point
├── Lincs_L1000.h5ad        # Data file
├── configs/                # Configuration files
│   ├── config.py           # Config dataclasses
│   └── base.yaml           # Default settings
├── checkpoints/            # Model checkpoints
│   ├── ldm/                # Diffusion model
│   └── vae/                # VAE encoder/decoder
├── cache/                  # Cached embeddings
├── src/                    # Source code
│   ├── data/               # Data loading
│   ├── models/             # Model architectures
│   ├── training/           # Training utilities
│   ├── evaluation/         # Evaluation metrics
│   └── utils/              # General utilities
└── scripts/                # Shell scripts
```

---

## Configuration

Modify `configs/base.yaml` to customize training:

```yaml
model:
  timesteps: 1000
  beta_schedule: linear
  latent_dim: 256

training:
  epochs: 200
  batch_size: 256
  max_lr: 0.001

data:
  data_path: ./Lincs_L1000.h5ad
  split_keys: [4foldcv_0, 4foldcv_1, 4foldcv_2, 4foldcv_3, 4foldcv_cell_0, 4foldcv_cell_1, 4foldcv_cell_2, 4foldcv_cell_3]
```

<!-- ---

## Citation

If you find this work useful, please cite:

```bibtex
@article{kim2025predicting,
  title={Predicting Condition-Aware Drug-Induced Transcriptional Responses via a Latent Diffusion Model},
  author={Kim, Chaewon and Yoo, Sunyong},
  journal={},
  year={2025}
}
``` -->

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Chaewon Kim: chaewonk215@gmail.com