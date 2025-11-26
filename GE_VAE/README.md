# Variational Autoencoder Pretraining

This module implements the variational autoencoder (VAE) used to construct the latent space for the Latent Diffusion Model (LDM) in our work, "Predicting Drug-Induced Transcriptional Responses Using Latent Diffusion Model."

The VAE is trained on perturbed gene expression (GE) profiles from the LINCS L1000 dataset and is responsible for mapping 978-dimensional expression vectors into a compact latent representation.
This latent space provides a stable and computationally efficient foundation for diffusion training.

## Train model
To train the GE_VAE, run the following command:
```
python train.py
```

You can adjust hyperparameters and settings in the `train.py` script as needed.
```python 
FILE_PATH = "../Lincs_L1000_mywrite_gene_fixed.h5ad"
SPLIT = "mysplit"
BATCH_SIZE = 1024
KLD_WEIGHT = 10
LATENT_EMB_DIM = 256
NORMALIZE = True
KFCV = 5
EPOCHS = 500
SCHEDULER = "exponential"
learning_rate = 0.001
scheduler_gamma = 0.99
```

## Pretrained Model
A pretrained GE_VAE model is uploaded at `GE_VAE/checkpoints/best_vae.pt`.