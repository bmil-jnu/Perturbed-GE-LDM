from collections import defaultdict
from logging import Logger
import os
import sys
import subprocess

import numpy as np
import torch
from run_training import run_training
from args import TrainArgs
from utils import create_logger, makedirs, timeit, set_seed

import scanpy as sc
from data.utils import train_valid_test


@timeit(logger_name='train_logger')
def cross_validate(rank, world_size, train_func, parallel=True):
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param rank: Rank of the current process.
    :param world_size: Total number of processes.
    :param train_func: Function which runs training.
    :param parallel: Whether to run in parallel (using multiple GPUs) or not.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    
    arguments = [
        '--epochs', '200',
        '--dataset_type', 'regression',
        '--data_path', './Lincs_L1000_mywrite_gene_fixed.h5ad',
        '--save_dir', 'checkpoints',
        '--comp_embed_model', 'molformer',
        '--metric', 'avg_gene_pearson',
    ]

    args = TrainArgs().parse_args(arguments)
    
    args.split_keys = ['4foldcv_0']
    
    args.rank = rank
    args.world_size = world_size
    args.batch_size = 256 * world_size
    args.loss_function = 'variational_loss' # 'variational_loss' or 'hybrid_loss' or 'mse_loss'
    args.lambda_kl = 0.001
    args.parallel = parallel
    
    args.save_smiles_splits = False
    args.save_preds = True
    args.train_data_size = 0
    args.data_sample = False
    args.save_intermediate = False
    
    args.mode = 'pred_mu_v' # 'pred_mu_var' or 'pred_mu_v'
    
    
    if not args.parallel:
        args.device = torch.device('cuda', 2)
    else:
        args.device = torch.device('cuda', rank)


    # Set up logger
    logger = create_logger(name='train_logger', save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    save_dir = args.save_dir

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    debug(f"Setting seeds...: {args.seed}")
    set_seed(args.seed)

    # Save args
    makedirs(args.save_dir)
    try:
        args.save(os.path.join(args.save_dir, 'args.json'))
    except subprocess.CalledProcessError:
        debug('Could not write the reproducibility section of the arguments to file, thus omitting this section.')
        args.save(os.path.join(args.save_dir, 'args.json'), with_reproducibility=False)

    # Get data
    debug('Loading Ann data...')
    adata = sc.read(args.data_path)
    
    args.adata_var_names = adata.var_names
    args.obs_key = 'cov_drug_name'
    args.task_names = args.adata_var_names

    # Run training on different random seeds for each fold
    all_scores = defaultdict(list)
    for index in range(len(args.split_keys)):
        
        split_key = args.split_keys[index]
        info(f'Fold {index}')
    
        args.save_dir = os.path.join(save_dir, f'fold_{index}')
        makedirs(args.save_dir)

        train_adata, valid_adata, test_adata, ctrl_len = train_valid_test(adata, split_key=split_key, sample=args.data_sample)
        debug(f'Number of data points = {len(train_adata) + len(valid_adata) + len(test_adata) - 3 * ctrl_len}')
        debug(f'--- Training data points = {len(train_adata) - ctrl_len}')
        debug(f'--- Validation data points = {len(valid_adata) - ctrl_len}')
        debug(f'--- Test data points = {len(test_adata) - ctrl_len}')
        
        debug(f'Number of tasks = {args.num_tasks}')
        
        args.train_data_size = len(train_adata)
        
        data = {
            'train': train_adata,
            'valid': valid_adata,
            'test': test_adata
            }
        
        model_scores = train_func(args, data, logger, split_key)


        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)
            
    all_scores = dict(all_scores)
    
    # Aggregate scores across folds
    mean_scores = {} 
    std_scores = {}
    for metric, scores in all_scores.items():
        mean_scores[metric] = np.mean(scores)
        std_scores[metric] = np.std(scores)
        
    info('Overall cross-validation results:')
    for metric in mean_scores.keys():
        info(f'{metric}: {mean_scores[metric]} ± {std_scores[metric]}')
        
    return mean_scores, std_scores
