import os
import json
from logging import Logger
from typing import Dict, List
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model import MoleculeModel
from models.nn_utils import param_count_all
from GE_VAE.model import GE_VAE
from data.Dataset import DrugDoseAnnDataset
from predict import predict
from utils import *

import wandb


def run_test(args, data, logger: Logger = None, split_key: str = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~args.TrainArgs` object containing arguments for
                 loading data and training the model.
    :param data: A :class:`~data.DrugDoseAnnDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """
    debug = info = print
    
    
    rank = args.rank
    world_size = args.world_size
    
    if args.parallel:
        setup(rank, world_size, args)
    
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    

    # Initialize wandb
    wandb.init(project="MiTCP", name=f"model_{args.model_idx}")
    wandb.config.update({
        'args': vars(args),
    }, allow_val_change=True)
    
    if args.parallel and dist.get_rank() != 0:  # Only log from rank 0
        wandb.log = debug
    
    debug("Test model {args.model_idx} on split {split_key}...")
    
    # Load data
    test_adata = data["test"]
    debug("Constructing AnnDatasets...")
    test_dataset = DrugDoseAnnDataset(test_adata, obs_key=args.obs_key)
    
    print("Length of test dataset:", len(test_dataset))
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True,
    )
    
    print("test_dataloader[0]:", test_dataloader.dataset.smiles_list[0])
    
    save_dir = os.path.join(args.save_dir, f'model_{args.model_idx}')
    makedirs(save_dir)
    
    
    debug(f"Building model {args.model_idx}")
    args.device = torch.device(f'cuda:{args.rank}' if torch.cuda.is_available() else 'cpu')
    model = MoleculeModel(args)

    debug(model)
    debug(f"Number of parameters = {param_count_all(model):,}")

    # Load VAE model
    debug("Loading GE_VAE model...")
    LATENT_EMB_DIM = 256  # Set the latent dimension size for the VAE
    ge_vae_path = "./GE_VAE/checkpoints/best_vae_no_meanlog.pt"
    wandb.config.update(
        {
            "vae_path": ge_vae_path,
            "latent_dim": LATENT_EMB_DIM,
        }
    )

    ge_vae = GE_VAE(input_dim=978, latent_dim=LATENT_EMB_DIM)
    ge_vae_state = torch.load(ge_vae_path, weights_only=True)
    ge_vae.load_state_dict(ge_vae_state["vae"])
    ge_vae.to(args.device)  # Move VAE to the same device as the model
    ge_vae.eval()  # Set the VAE to evaluation mode
    
    for param in ge_vae.parameters():
        param.requires_grad = False

    model.train()
    ge_vae.eval()
    
    
    model, scaler = load_checkpoint_for_test(f'./checkpoints/fold_0/model_{args.model_idx}/model.pt', device=args.device, logger=logger)
    model = model.to(args.device)
    

    test_scores, test_preds = None, None
    test_scores, test_preds = predict(
        model=model,
        ge_vae=ge_vae,
        data_loader=test_dataloader,
        scaler=scaler,
        args=args,
        test=True,
        debug=debug,
        dist_enabled=False,
    )

    # Save scores
    save_dir = os.path.join(save_dir, split_key)
    makedirs(save_dir)
    
    with open(os.path.join(save_dir, "test_scores.json"), "w") as f:
        json.dump(str(test_scores), f, indent=4, sort_keys=True)

    test_cov_drug_list = test_dataset.obs_list

    with open(os.path.join(save_dir, 'cov_drug_array.csv'), "a+") as f:
        for i in test_cov_drug_list:
            f.write(i + "\n")

    # Optionally save test preds
    if args.save_preds:
        test_preds_dataframe = pd.DataFrame(
            data={
                "smiles": test_dataset.smiles_list,
                "cell_id": test_dataset.obs_list,
                "dose": test_dataset.dose_tensor.view(-1).tolist(),
                "time": test_dataset.time_tensor.view(-1).tolist(),
            }
        )

        all_preds = []

        for preds in test_preds:  # list of (batch_size, num_tasks)
            df = pd.DataFrame(data=preds, columns=args.task_names)
            all_preds.append(df)

        value_df = pd.concat(all_preds, axis=0, ignore_index=True)

        test_preds_dataframe = pd.concat(
            [test_preds_dataframe.reset_index(drop=True), value_df], axis=1
        )

        try:
            test_preds_dataframe.to_csv(
                os.path.join(save_dir, "test_preds.csv"), index=False
            )
        except Exception as e:
            debug(f"Warning: Could not save test predictions to CSV due to {e}.")
            test_preds.to_csv(
                'test_preds.csv', index=False
            )
            
        test_truth = test_dataset.data.numpy()
        test_truth_df = pd.DataFrame(data=test_truth, columns=args.task_names)
        test_truth_df.to_csv(
            os.path.join(save_dir, "test_truth.csv"), index=False
        )

    return test_scores
