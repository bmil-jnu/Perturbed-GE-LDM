#!/usr/bin/env python
"""
LDM-LINCS: Main entry point for training and evaluation.

This is the single entry point for all operations:
- Training: python main.py train --config configs/base.yaml
- Evaluation: python main.py eval --config configs/base.yaml
- Prediction: python main.py predict --config configs/base.yaml

Examples:
    # Train with default config
    python main.py train
    
    # Train with custom config
    python main.py train --config configs/experiment.yaml
    
    # Train with config overrides
    python main.py train --epochs 100 --batch_size 512
    
    # Evaluate trained model
    python main.py eval --checkpoint checkpoints/fold_0/model.pt
    
    # Run distributed training
    python main.py train --parallel --world_size 4
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict

import torch
import torch.multiprocessing as mp

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import ExperimentConfig, load_config
from src.data import LINCSDataModule
from src.models import ModelFactory
from src.training import Trainer, DistributedManager, get_loss_func, build_optimizer_and_scheduler
from src.evaluation import Evaluator, Predictor
from src.utils import create_logger, set_seed, makedirs

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LDM-LINCS: Latent Diffusion Model for Gene Expression Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    _add_common_args(train_parser)
    _add_training_args(train_parser)
    
    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    
    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    _add_common_args(predict_parser)
    predict_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    predict_parser.add_argument("--output", type=str, default="predictions.csv", help="Output file path")
    
    args = parser.parse_args()
    
    # Default to train mode if not specified
    if args.mode is None:
        args.mode = "train"
    
    return args


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser."""
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Path to AnnData file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments."""
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--parallel", action="store_true", help="Use distributed training")
    parser.add_argument("--world_size", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--split_keys", type=str, nargs="+", help="Split keys for cross-validation")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_run_id", type=str, help="WandB run ID to resume logging")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build configuration from args and config file."""
    # Load base config if provided
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = ExperimentConfig()
    
    # Override with command line args
    if hasattr(args, 'data_path') and args.data_path:
        config.data.data_path = args.data_path
    if hasattr(args, 'save_dir') and args.save_dir:
        config.save_dir = args.save_dir
    if hasattr(args, 'seed'):
        config.training.seed = args.seed
    if hasattr(args, 'device'):
        config.device = args.device
    if hasattr(args, 'quiet'):
        config.quiet = args.quiet
    if hasattr(args, 'no_wandb') and args.no_wandb:
        config.use_wandb = False
    
    # Training specific
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
    if hasattr(args, 'lr') and args.lr:
        config.training.max_lr = args.lr
    if hasattr(args, 'parallel') and args.parallel:
        config.training.parallel = True
    if hasattr(args, 'split_keys') and args.split_keys:
        config.data.split_keys = args.split_keys
    
    # Resume settings
    if hasattr(args, 'resume') and args.resume:
        config.resume = True
    else:
        config.resume = getattr(config, 'resume', False)
    if hasattr(args, 'resume_checkpoint') and args.resume_checkpoint:
        config.resume_checkpoint = args.resume_checkpoint
    if hasattr(args, 'wandb_run_id') and args.wandb_run_id:
        config.wandb_run_id = args.wandb_run_id
    
    # Set mode
    config.mode = args.mode
    
    return config


def run_training_fold(
    rank: int,
    world_size: int,
    config: ExperimentConfig,
    split_key: str,
    fold_idx: int,
) -> Dict[str, float]:
    """
    Run training for a single fold.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Experiment configuration
        split_key: Data split key
        fold_idx: Fold index
        
    Returns:
        Dictionary of test metrics
    """
    # Setup distributed
    dist_manager = DistributedManager(world_size=world_size)
    if config.training.parallel:
        dist_manager.setup(rank)
    
    device = dist_manager.device if config.training.parallel else torch.device(config.device)
    is_main = dist_manager.is_main_process
    
    # Setup logging
    fold_save_dir = os.path.join(config.save_dir, split_key)
    makedirs(fold_save_dir)
    
    logger = create_logger(
        name=f"train_{split_key}",
        save_dir=fold_save_dir,
        quiet=config.quiet
    )
    
    if is_main:
        logger.info(f"Starting fold {fold_idx}: {split_key}")
        logger.info(f"Device: {device}")
    
    # Initialize wandb
    if config.use_wandb and WANDB_AVAILABLE and is_main:
        wandb_kwargs = {
            "project": config.wandb_project,
            "name": f"{split_key}_fold_{fold_idx}",
            "group": f"experiment_{config.experiment_name}",
            "config": config.to_dict(),
        }
        # Resume wandb run if specified
        if getattr(config, 'resume', False) and getattr(config, 'wandb_run_id', None):
            wandb_kwargs["id"] = config.wandb_run_id
            wandb_kwargs["resume"] = "must"
            logger.info(f"Resuming wandb run: {config.wandb_run_id}")
        wandb.init(**wandb_kwargs)
    
    # Load data
    datamodule = LINCSDataModule(
        data_path=config.data.data_path,
        split_key=split_key,
        obs_key=config.data.obs_key,
        cache_dir=config.data.cache_dir,
        data_sample=config.data.data_sample,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    datamodule.setup(device=device)
    
    if is_main:
        logger.info(f"Train size: {datamodule.train_size}")
        logger.info(f"Val size: {datamodule.val_size}")
        logger.info(f"Test size: {datamodule.test_size}")
    
    # Create models
    model, vae = ModelFactory.create_models(
        device=device,
        model_config={
            "timesteps": config.model.timesteps,
            "beta_schedule": config.model.beta_schedule,
            "mode": config.model.mode,
            "latent_dim": config.model.latent_dim,
            "hidden_dim": config.model.hidden_dim,
            "ge_dim": config.model.ge_dim,
            "vae_path": config.model.vae_path,
            "vae_latent_dim": config.model.vae_latent_dim,
        }
    )
    
    # Wrap model for distributed
    if config.training.parallel:
        model = dist_manager.wrap_model(model)
    
    # Build optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        train_data_size=datamodule.train_size,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        warmup_epochs=config.training.warmup_epochs,
        init_lr=config.training.init_lr,
        max_lr=config.training.max_lr,
        final_lr=config.training.final_lr,
    )
    
    # Get loss function
    loss_fn = get_loss_func(config.training.loss_function)
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        vae=vae,
        device=device,
        n_steps=config.model.n_steps,
        mode=config.model.mode,
        distributed=config.training.parallel,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        vae=vae,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device,
        distributed=config.training.parallel,
        rank=rank,
        world_size=world_size,
        use_wandb=config.use_wandb and WANDB_AVAILABLE,
        evaluator=evaluator,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if getattr(config, 'resume', False):
        checkpoint_path = getattr(config, 'resume_checkpoint', None)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(fold_save_dir, "model.pt")
        
        if os.path.exists(checkpoint_path):
            if is_main:
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            start_epoch = trainer.load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if is_main:
                logger.info(f"Resuming from epoch {start_epoch}")
        else:
            if is_main:
                logger.warning(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")
    
    # Create data loaders
    train_loader = datamodule.train_dataloader(
        batch_size=config.training.batch_size,
        distributed=config.training.parallel,
        rank=rank,
        world_size=world_size,
        seed=config.training.seed,
    )
    
    val_loader = datamodule.val_dataloader(
        batch_size=config.training.batch_size,
        distributed=config.training.parallel,
        rank=rank,
        world_size=world_size,
    )
    
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.epochs,
        save_dir=fold_save_dir,
        early_stop_patience=config.training.early_stop_patience,
        metric=config.training.metric,
        start_epoch=start_epoch,
    )
    
    # Evaluate on test set
    test_loader = datamodule.test_dataloader(batch_size=config.training.batch_size)
    
    test_metrics = {}
    if is_main:
        # Load best model
        best_model_path = os.path.join(fold_save_dir, "model.pt")
        if os.path.exists(best_model_path):
            model, _, _ = ModelFactory.load_checkpoint(
                best_model_path,
                device=device,
            )
            evaluator.model = model
        
        test_metrics = evaluator.evaluate(test_loader)
        
        logger.info(f"Test metrics for {split_key}:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        # Save predictions if requested
        if config.training.save_preds:
            predictor = Predictor(
                model=model,
                vae=vae,
                device=device,
                n_steps=config.model.n_steps,
                mode=config.model.mode,
                task_names=list(datamodule.var_names),
            )
            
            preds, metadata = predictor.predict(test_loader, return_metadata=True)
            predictor.save_predictions(
                preds,
                os.path.join(fold_save_dir, "test_predictions.csv"),
                metadata,
            )
    
    # Cleanup
    if config.training.parallel:
        dist_manager.cleanup()
    
    if config.use_wandb and WANDB_AVAILABLE and is_main:
        wandb.finish()
    
    return test_metrics


def cross_validate(
    rank: int,
    world_size: int,
    config: ExperimentConfig,
) -> None:
    """
    Run cross-validation training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        config: Experiment configuration
    """
    set_seed(config.training.seed)
    
    is_main = (rank == 0)
    all_scores = defaultdict(list)
    
    for fold_idx, split_key in enumerate(config.data.split_keys):
        if is_main:
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{len(config.data.split_keys)}: {split_key}")
            print(f"{'='*50}\n")
        
        fold_scores = run_training_fold(
            rank=rank,
            world_size=world_size,
            config=config,
            split_key=split_key,
            fold_idx=fold_idx,
        )
        
        for metric, value in fold_scores.items():
            all_scores[metric].append(value)
    
    # Print aggregate results
    if is_main and all_scores:
        print("\n" + "="*50)
        print("Cross-Validation Results")
        print("="*50)
        
        import numpy as np
        for metric, scores in all_scores.items():
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"{metric}: {mean:.4f} ± {std:.4f}")


def run_evaluation(config: ExperimentConfig, checkpoint_path: str) -> Dict[str, float]:
    """
    Run evaluation on test set.
    
    Args:
        config: Experiment configuration
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Dictionary of metrics
    """
    device = torch.device(config.device)
    set_seed(config.training.seed)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load model
    model, scaler, state = ModelFactory.load_checkpoint(checkpoint_path, device=device)
    
    # Create VAE
    vae = ModelFactory.create_vae(
        device=device,
        checkpoint_path=config.model.vae_path,
    )
    
    # Load data
    split_key = config.data.split_keys[0]
    datamodule = LINCSDataModule(
        data_path=config.data.data_path,
        split_key=split_key,
        obs_key=config.data.obs_key,
        cache_dir=config.data.cache_dir,
    )
    datamodule.setup(device=device)
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        vae=vae,
        device=device,
        n_steps=config.model.n_steps,
        mode=config.model.mode,
        scaler=scaler,
    )
    
    # Evaluate
    test_loader = datamodule.test_dataloader(batch_size=config.training.batch_size)
    metrics = evaluator.evaluate(test_loader)
    
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics


def run_prediction(
    config: ExperimentConfig,
    checkpoint_path: str,
    output_path: str,
) -> None:
    """
    Run prediction and save results.
    
    Args:
        config: Experiment configuration
        checkpoint_path: Path to model checkpoint
        output_path: Path to save predictions
    """
    device = torch.device(config.device)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load model
    model, scaler, _ = ModelFactory.load_checkpoint(checkpoint_path, device=device)
    
    # Create VAE
    vae = ModelFactory.create_vae(
        device=device,
        checkpoint_path=config.model.vae_path,
    )
    
    # Load data
    split_key = config.data.split_keys[0]
    datamodule = LINCSDataModule(
        data_path=config.data.data_path,
        split_key=split_key,
        obs_key=config.data.obs_key,
        cache_dir=config.data.cache_dir,
    )
    datamodule.setup(device=device)
    
    # Create predictor
    predictor = Predictor(
        model=model,
        vae=vae,
        device=device,
        n_steps=config.model.n_steps,
        mode=config.model.mode,
        scaler=scaler,
        task_names=list(datamodule.var_names),
    )
    
    # Predict
    test_loader = datamodule.test_dataloader(batch_size=config.training.batch_size)
    predictions, metadata = predictor.predict(test_loader, return_metadata=True)
    
    # Save
    predictor.save_predictions(predictions, output_path, metadata)
    print(f"Predictions saved to {output_path}")


def main():
    """Main entry point."""
    start_time = datetime.now()
    
    # Parse arguments
    args = parse_args()
    
    # Build configuration
    config = build_config(args)
    
    print(f"LDM-LINCS: {args.mode.upper()} mode")
    print(f"Config: {args.config or 'default'}")
    print(f"Save dir: {config.save_dir}")
    
    # Set seed
    set_seed(config.training.seed)
    
    if args.mode == "train":
        # Determine world size
        world_size = getattr(args, 'world_size', None)
        if world_size is None:
            world_size = torch.cuda.device_count() if config.training.parallel else 1
        
        config.world_size = world_size
        
        if config.training.parallel and world_size > 1:
            print(f"Starting distributed training with {world_size} GPUs")
            mp.spawn(
                cross_validate,
                args=(world_size, config),
                nprocs=world_size,
            )
        else:
            cross_validate(rank=0, world_size=1, config=config)
    
    elif args.mode == "eval":
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            # Try to find latest checkpoint
            checkpoint_path = os.path.join(config.save_dir, "model.pt")
        
        run_evaluation(config, checkpoint_path)
    
    elif args.mode == "predict":
        run_prediction(config, args.checkpoint, args.output)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print(f"\nCompleted in {duration:.2f} minutes")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")


if __name__ == "__main__":
    main()
