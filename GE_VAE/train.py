import os
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torchinfo import summary
import scanpy as sc

from model import GE_VAE
from utils import set_seed, GeneDataset, loss_function, rowwise_pcc, validate
import wandb
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


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


class GE_VAE_Trainer:
    def __init__(self):

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.adata = sc.read_h5ad(FILE_PATH)
        set_seed()

        self.rmse_kfold_list = []
        self.mse_kfold_list = []
        self.test_loss_kfold_list = []
        self.pccs_gene_kfold_list = []
        self.r2s_gene_kfold_list = []
        self.pccs_row_kfold_list = []
        self.r2s_row_kfold_list = []

    def make_k_fold_dataset(self, split_num):

        self.split_num = split_num

        self.split_col = f"{SPLIT}_{split_num}"

        # create output directory
        self.output_dir = f"checkpoints/{self.split_col}"
        os.makedirs(self.output_dir, exist_ok=True)

        # define model_weights path including best weights
        self.model_path = os.path.join(self.output_dir, "best_vae.pt")

        train_data = self.adata.X[self.adata.obs[self.split_col] == "train"]
        valid_data = self.adata.X[self.adata.obs[self.split_col] == "valid"]
        test_data = self.adata.X[self.adata.obs[self.split_col] == "test"]

        print(
            f"The shape of train_data: {train_data.shape}, valid_data: {valid_data.shape}, test_data: {test_data.shape}"
        )

        train_dataset = GeneDataset(train_data)
        valid_dataset = GeneDataset(valid_data)
        test_dataset = GeneDataset(test_data)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )

        print(
            f"The length of train_loader: {len(self.train_loader)}, valid_loader: {len(self.valid_loader)}, test_loader: {len(self.test_loader)}"
        )

    def train_k_fold(self):

        # initialize the best validation loss as infinity
        self.best_val_loss = float("inf")
        self.es_patience = 20
        self.patience_counter = 0

        model = GE_VAE(input_dim=978, latent_dim=LATENT_EMB_DIM)

        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if SCHEDULER == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=scheduler_gamma
            )
        elif SCHEDULER == "reduce":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10, verbose=True
            )

        run = wandb.init(
            # Set the wandb project where this run will be logged.
            project="GE_VAE",
            # Set the wandb run name.
            name=self.split_col,
            # Track hyperparameters and run metadata.
            config={
                "model": model.__class__.__name__,
                "batch_size": BATCH_SIZE,
                "latent_emb_dim": LATENT_EMB_DIM,
                "kld_weight": KLD_WEIGHT,
                "loss_function": loss_function.__name__,
                "optimizer": "Adam",
                "scheduler": "ExponentialLR",
                "learning_rate": learning_rate,
                "scheduler_gamma": scheduler_gamma,
                # "train_dataset": train_datapath,
                "epochs": EPOCHS,
                "early_stopping_patience": self.es_patience,
                "Normalize": NORMALIZE,
                "activation_function": "LeakyReLU",
            },
        )

        print("Training Started!!")
        print("Model architecture:")
        print(
            summary(
                model,
                input_size=(BATCH_SIZE, 978),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                depth=3,
            )
        )

        model = model.to(self.device)

        # start training by looping over the number of epochs
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            running_loss = 0.0
            running_recon_loss = 0.0
            running_kld_loss = 0.0

            for i, x in enumerate(self.train_loader):
                x = x.to(self.device)
                optimizer.zero_grad()
                predictions = model(x)
                total_loss = loss_function(predictions, KLD_WEIGHT)

                # Backward pass
                total_loss["loss"].backward()

                # Optimizer variable updates
                optimizer.step()

                running_loss += total_loss["loss"].item()
                running_recon_loss += total_loss["Reconstruction_Loss"].item()
                running_kld_loss += total_loss["KLD"].item()

            # compute average loss for the epoch
            train_loss = running_loss / len(self.train_loader)
            train_recon_loss = running_recon_loss / len(self.train_loader)
            train_kld_loss = running_kld_loss / len(self.train_loader)

            # compute validation loss for the epoch
            val_loss_dict = validate(
                model, self.valid_loader, self.device, loss_function, KLD_WEIGHT
            )
            val_loss = val_loss_dict["val_loss"]
            val_pccs_row = val_loss_dict["pcc_row_mean"]

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss}"
                )
                print(
                    f"    Reconstruction Loss: {train_recon_loss:.4f}, KLD Loss: {train_kld_loss:.4f}"
                )

            # save best vae model weights based on validation loss
            with open("./checkpoints/train.log", "a", encoding="utf-8") as f:
                f.write(
                    f"Epoch {epoch+1}/{EPOCHS}\n"
                    + f"    Total Loss: {total_loss['loss'].detach().item():.4f}\n"
                    + f"    Reconstruction Loss: {total_loss['Reconstruction_Loss']:.4f}\n"
                    + f"    KL Divergence Loss: {total_loss['KLD']:.4f}\n"
                    + f"    Val Loss: {val_loss}\n"
                    + f"    Val PCCs: {val_pccs_row}\n"
                )

                # Early Stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save({"vae": model.state_dict()}, self.model_path)
                    f.write(f"[EPOCH: {epoch+1}] SAVE BEST MODEL!!\n")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    f.write(
                        f"[EPOCH: {epoch+1}] No improvement in validation loss. Current patience: {self.patience_counter}/{self.es_patience}\n"
                    )
                    if self.patience_counter >= self.es_patience:
                        f.write(
                            f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.\n"
                        )
                        # break

            if SCHEDULER == "reduce":
                # Step the scheduler based on validation loss
                scheduler.step(val_loss)
            elif SCHEDULER == "exponential":
                # Step the scheduler every epoch
                scheduler.step()

            run.log(
                {
                    "Epoch": epoch + 1,
                    "Train/train_loss": train_loss,
                    "Train/reconstruction_loss": total_loss["Reconstruction_Loss"]
                    .detach()
                    .item(),
                    "Train/kld_loss": total_loss["KLD"].detach().item(),
                    "Validation/val_loss": val_loss,
                    "Validation/learning_rate": optimizer.param_groups[0]["lr"],
                    "Validation/val_pccs_row": val_pccs_row,
                }
            )

        print("Finished Training!")

    def evaluate_k_fold(self):

        model = GE_VAE(input_dim=978, latent_dim=LATENT_EMB_DIM)

        # model load
        model_state = torch.load(self.model_path, weights_only=True)
        model.load_state_dict(model_state["vae"])
        model.to(self.device)
        model.eval()

        test_loss = 0.0
        all_recons = []
        all_inputs = []

        with torch.no_grad():
            for x in self.test_loader:
                x = x.to(self.device)
                predictions = model(x)
                recons = predictions[0]

                all_recons.append(recons.cpu())
                all_inputs.append(x.cpu())

                total_loss = loss_function(predictions, KLD_WEIGHT)
                test_loss += total_loss["loss"].item()

        test_loss /= len(self.test_loader)

        all_recons = torch.cat(all_recons, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)

        pccs_gene = []
        for i in range(all_inputs.shape[1]):
            input_gene = all_inputs[:, i]
            recon_gene = all_recons[:, i]
            pcc = torch.corrcoef(torch.stack([input_gene, recon_gene]))[0, 1]
            pccs_gene.append(pcc.item())
        pccs_gene = np.mean(pccs_gene)

        r2s_gene = []
        for i in range(all_inputs.shape[1]):
            input_gene = all_inputs[:, i]
            recon_gene = all_recons[:, i]
            r2 = r2_score(input_gene.numpy(), recon_gene.numpy())
            r2s_gene.append(r2)

        pccs_row = []
        for i in range(all_inputs.shape[1]):
            input_row = all_inputs[i, :]
            recon_row = all_recons[i, :]

            pcc = np.corrcoef(input_row.numpy(), recon_row.numpy())[0, 1]
            pccs_row.append(pcc)

        r2s_row = []
        for i in range(all_inputs.shape[1]):
            input_row = all_inputs[i, :]
            recon_row = all_recons[i, :]
            r2 = r2_score(input_row.numpy(), recon_row.numpy())
            r2s_row.append(r2)

        rmse = root_mean_squared_error(all_inputs.numpy(), all_recons.numpy())
        mse = mean_squared_error(all_inputs.numpy(), all_recons.numpy())

        self.rmse_kfold_list.append(rmse)
        self.mse_kfold_list.append(mse)
        self.test_loss_kfold_list.append(test_loss)
        self.pccs_gene_kfold_list.append(np.mean(pccs_gene))
        self.r2s_gene_kfold_list.append(np.mean(r2s_gene))
        self.pccs_row_kfold_list.append(np.mean(pccs_row))
        self.r2s_row_kfold_list.append(np.mean(r2s_row))

        print(f"Evaluation on Test Set for {self.split_col}:")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MSE: {mse:.4f}")
        print(f"    Test Loss: {test_loss:.4f}")
        print(f"    Average PCC: {np.mean(pccs_gene):.4f}")
        print(f"    Average R2: {np.mean(r2s_gene):.4f}")
        print(f"    Average PCC (row): {np.mean(pccs_row):.4f}")
        print(f"    Average R2 (row): {np.mean(r2s_row):.4f}")

        wandb.log(
            {
                "Test/RMSE": rmse,
                "Test/MSE": mse,
                "Test/Test Loss": test_loss,
                "Test/Average PCC": np.mean(pccs_gene),
                "Test/Average R2": np.mean(r2s_gene),
                "Test/Average PCC (row)": np.mean(pccs_row),
                "Test/Average R2 (row)": np.mean(r2s_row),
            }
        )

        with open("./checkpoints/train.log", "a", encoding="utf-8") as f:
            f.write(f"Evaluation on Test Set for {self.split_col}:\n")
            f.write(f"    RMSE: {rmse:.4f}\n")
            f.write(f"    MSE: {mse:.4f}\n")
            f.write(f"    Test Loss: {test_loss:.4f}\n")
            f.write(f"    Average PCC: {np.mean(pccs_gene):.4f}\n")
            f.write(f"    Average R2: {np.mean(r2s_gene):.4f}\n")
            f.write(f"    Average PCC (row): {np.mean(pccs_row):.4f}\n")
            f.write(f"    Average R2 (row): {np.mean(r2s_row):.4f}\n")

            f.write("\n")
            f.write(f"Finished evaluation for {self.split_col}.\n")

    def run_k_fold_cv(self, k_fold_num=5):
        for split_num in range(0, k_fold_num):
            print(f"Starting K-Fold CV for split: {split_num}")
            os.makedirs("checkpoints", exist_ok=True)
            with open("./checkpoints/train.log", "a", encoding="utf-8") as f:
                f.write(f"\nStarting K-Fold CV for split: {split_num}\n")
            self.make_k_fold_dataset(split_num)
            self.train_k_fold()
            self.evaluate_k_fold()

        print(f"K-Fold CV Results after {k_fold_num} folds:")

        print(f"    Average RMSE: {np.mean(self.rmse_kfold_list):.4f} ± {np.std(self.rmse_kfold_list):.4f}")
        print(f"    Average MSE: {np.mean(self.mse_kfold_list):.4f} ± {np.std(self.mse_kfold_list):.4f}")
        print(f"    Average Test Loss: {np.mean(self.test_loss_kfold_list):.4f} ± {np.std(self.test_loss_kfold_list):.4f}")
        print(f"    Average PCC (gene): {np.mean(self.pccs_gene_kfold_list):.4f} ± {np.std(self.pccs_gene_kfold_list):.4f}")
        print(f"    Average R2 (gene): {np.mean(self.r2s_gene_kfold_list):.4f} ± {np.std(self.r2s_gene_kfold_list):.4f}")
        print(f"    Average PCC (row): {np.mean(self.pccs_row_kfold_list):.4f} ± {np.std(self.pccs_row_kfold_list):.4f}")
        print(f"    Average R2 (row): {np.mean(self.r2s_row_kfold_list):.4f} ± {np.std(self.r2s_row_kfold_list):.4f}")


if __name__ == "__main__":
    trainer = GE_VAE_Trainer()
    trainer.run_k_fold_cv(k_fold_num=5)