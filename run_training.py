import json
from logging import Logger
import os
from typing import Dict, List

import numpy as np
import random
import pandas as pd
from tqdm import trange

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from predict import predict
from train import train
from loss_functions import get_loss_func

from models.model import MoleculeModel
from models.nn_utils import param_count_all
from utils import *
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from GE_VAE.VAE_model import GE_VAE

import wandb

from data.Dataset import DrugDoseAnnDataset




def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def run_training(args, data, logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~args.TrainArgs` object containing arguments for
                 loading data and training the model.
    :param data: A :class:`~data.DrugDoseAnnDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """

    debug = info = print


    # Initialize distributed training
    rank = args.rank
    world_size = args.world_size
    setup(rank, world_size)

    # Set model index
    if os.path.exists(args.save_dir):
        dirs = [
            d
            for d in os.listdir(args.save_dir)
            if os.path.isdir(os.path.join(args.save_dir, d))
        ]
        model_idx = len(dirs)
    else:
        model_idx = 0
        
    # Save file names
    MODEL_FILE_NAME = "model.pt"

    # Initialize wandb
    # TODO: Logging only from rank 0
    wandb.init(project="MiTCP", name=f"model_{model_idx}", group="DDP")
    wandb.config.update(
        {
            "data_path": args.data_path,
            "loss_function": args.loss_function,
            "args": vars(args),
        }
    )
    if dist.get_rank() != 0:  # Only log from rank 0
        wandb.log = debug

    # Get loss function
    loss_func = get_loss_func(args)

    # Load data
    train_adata = data["train"]
    valid_adata = data["valid"]
    test_adata = data["test"]

    debug("Constructing AnnDatasets...")
    train_dataset = DrugDoseAnnDataset(train_adata, obs_key=args.obs_key)
    valid_dataset = DrugDoseAnnDataset(valid_adata, obs_key=args.obs_key)
    test_dataset = DrugDoseAnnDataset(test_adata, obs_key=args.obs_key)

    # debug("Fitting targets scaler for train_data")
    # scaler = train_dataset.normalize_targets()
    # valid_dataset.set_targets(scaler.transform(valid_dataset.data))
    # test_dataset.set_targets(scaler.transform(test_dataset.data))
    scaler = None

    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )
    valid_sampler = DistributedSampler(
        dataset=valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4, pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, sampler=valid_sampler, batch_size=args.batch_size, num_workers=4, pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    print("train_data_loader[0]:", train_dataloader.dataset.smiles_list[0])

    save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
    makedirs(save_dir)

        
    debug(f"Building model {model_idx}")
    model = MoleculeModel(args)

    debug(model)
    debug(f"Number of parameters = {param_count_all(model):,}")

    if args.cuda:
        debug("Moving model to cuda")

    if args.parallel:
        model.to(rank)
        model = DDP(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=False
        )
    else:
        model = model.to(args.device)

    # Optimizers
    optimizer = build_optimizer(model, args)
    wandb.config.update(
        {
            "optimizer": optimizer.__class__.__name__,
            "optimizer_params": optimizer.__dict__,
        }
    )

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)
    # import math
    # steps_per_epoch = math.ceil(len(train_dataloader) / (world_size))
    # total_steps = args.epochs * steps_per_epoch  # epochs=500
    # warmup_steps = min(5000, int(0.03 * total_steps))
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    wandb.config.update(
        {
            "scheduler": scheduler.__class__.__name__,
            "scheduler_params": scheduler.__dict__,
        }
    )
    
    # Ensure that model is saved in correct location for evaluation if 0 epochs
    save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, args, optimizer, scheduler, epoch=0)

    # Run training
    EARLY_STOP_CRITERION = args.early_stop_criterion
    early_stop_count = 0
    best_score = -float("inf")
    best_epoch, n_iter = 0, 0

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

    for epoch in trange(args.epochs):
        debug(f"Epoch {epoch}")
        wandb.log({"Epoch": epoch})

        if args.parallel:
            train_dataloader.sampler.set_epoch(epoch)  # Shuffle data at each epoch
            valid_dataloader.sampler.set_epoch(epoch)

        n_iter = train(
            model=model,
            ge_vae=ge_vae,
            data_loader=train_dataloader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
            logger=logger,
        )

        if isinstance(scheduler, ExponentialLR):
            scheduler.step()

        val_scores_all_ranks, _ = predict(
            model=model,
            ge_vae=ge_vae,
            data_loader=valid_dataloader,
            scaler=scaler,
            args=args,
            debug=debug,
        )


        args.metric = "avg_gene_pearson"

        # Save model checkpoint if improved validation score
        if not dist.is_initialized() or dist.get_rank() == 0:
            cov_drug_list = valid_dataset.obs_list
            
            wandb.config.update(
                {
                    "es_metric": args.metric,
                }
            )
            mean_val_score = val_scores_all_ranks[args.metric]

            if mean_val_score > best_score:
                best_score, best_epoch = mean_val_score, epoch
                early_stop_count = 0
                save_checkpoint(
                    os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, args, optimizer, scheduler, epoch=epoch
                )
            else:
                early_stop_count += 1
                debug(
                    f"No improvement on validation {args.metric} for {early_stop_count} epochs."
                )
                if early_stop_count >= EARLY_STOP_CRITERION:
                    break

    if dist.is_initialized():
        rank = dist.get_rank()
        dist_enabled = True
    else:
        rank = 0
        dist_enabled = False

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")

    # Evaluate on test set using model with best validation score
    info(
        f"Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}"
    )
    model, scaler = load_checkpoint(
        os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger
    )

    test_scores, test_preds = None, None
    if not dist.is_initialized() or dist.get_rank() == 0:
        test_scores, test_preds = predict(
            model=model,
            ge_vae=ge_vae,
            data_loader=test_dataloader,
            scaler=scaler,
            args=args,
            test=True,
            debug=debug,
            dist_enabled=dist_enabled,
        )

    # Save scores
    with open(os.path.join(save_dir, "test_scores.json"), "w") as f:
        json.dump(str(test_scores), f, indent=4, sort_keys=True)

    with open(save_dir + args.split_key + "_cov_drug_array.csv", "a+") as f:
        for i in cov_drug_list:
            f.write(i + "\n")

    # Optionally save test preds
    if args.save_preds:
        test_preds_dataframe = pd.DataFrame(
            data={
                "smiles": test_dataset.smiles_list,
                "cell_id": test_dataset.obs_list,
                "dose": test_dataset.dose_list,
                "time": test_dataset.time_list,
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

        test_preds_dataframe.to_csv(
            os.path.join(save_dir, "test_preds.csv"), index=False
        )

    return test_scores
