from argparse import Namespace

import datetime
from datetime import timedelta
from functools import wraps
import logging
import os
from time import time
import random
from typing import Any, Callable, List, Tuple, Union, Optional
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from args import TrainArgs
from data.utils import StandardScaler
from models.model import MoleculeModel
from models.nn_utils import NoamLR

from torch.autograd import Variable
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(days=2)
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    scaler: Optional[StandardScaler] = None,
    args: Optional[TrainArgs] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: Optional[int] = None,
    best_score: Optional[float] = None,
    early_stop_count: Optional[int] = None,
) -> None:
    if args is not None:
        args = Namespace(**args.as_dict())

    data_scaler = {"means": scaler.means, "stds": scaler.stds} if scaler else None

    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_score": best_score,
        "early_stop_count": early_stop_count,
        "random_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    scaler: Optional[StandardScaler] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Union[torch.device, str] = "cpu",
    logger: logging.Logger = None,
) -> Tuple[
    torch.nn.Module,
    Optional[StandardScaler],
    Optional[Optimizer],
    Optional[_LRScheduler],
    int,
    Optional[float],
    Optional[int],
]:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param model: The model to load the checkpoint into.
    :param scaler: The data scaler to load the checkpoint into.
    :param optimizer: The optimizer to load the checkpoint into.
    :param scheduler: The learning rate scheduler to load the checkpoint into.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded model, scaler, optimizer, scheduler, epoch, best score, and early stop count.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(
        path, map_location=lambda storage, loc: storage, weights_only=False
    )

    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    loaded_state_dict = state["state_dict"]

    if device is not None:
        args.device = device

    # Load pretrained weights
    model_state_dict = model.state_dict()

    for key in list(loaded_state_dict.keys()):
        if (
            key not in model_state_dict
            or loaded_state_dict[key].shape != model_state_dict[key].shape
        ):
            info(
                f'Warning: Pretrained parameter "{key}" cannot be loaded due to size mismatch or missing key.'
            )
            loaded_state_dict.pop(key)

    # Load pretrained weights
    model_state_dict.update(loaded_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)

    # Load scaler
    if scaler is not None and state["data_scaler"] is not None:
        scaler.means = state["data_scaler"]["means"]
        scaler.stds = state["data_scaler"]["stds"]

    # Load optimizer
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])

    # Load scheduler
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    # Load random state for reproducibility
    if "random_state" in state:
        random_state = state["random_state"]
        torch.set_rng_state(random_state["torch"])
        torch.cuda.set_rng_state_all(random_state["cuda"])
        np.random.set_state(random_state["numpy"])
        random.setstate(random_state["random"])
    return (
        model,
        scaler,
        optimizer,
        scheduler,
        state.get("epoch", 0),
        state.get("best_score", None),
        state.get("early_stop_count", None),
    )


def load_checkpoint_for_test(
    path: str, device: torch.device = None, logger: logging.Logger = None
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(
        path, map_location=lambda storage, loc: storage, weights_only=False
    )

    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    loaded_state_dict = state["state_dict"]

    data_scaler = state["data_scaler"]

    if device is not None:
        args.device = device

    # Build model
    model = MoleculeModel(args)
    model_state_dict = model.state_dict()

    for key in list(loaded_state_dict.keys()):
        if key.startswith("module."):
            new_key = key[7:]  # Remove 'module.' prefix
            loaded_state_dict[new_key] = loaded_state_dict.pop(key)

    for key in list(loaded_state_dict.keys()):
        if (
            key not in model_state_dict
            or loaded_state_dict[key].shape != model_state_dict[key].shape
        ):
            info(
                f'Warning: Pretrained parameter "{key}" cannot be loaded due to size mismatch or missing key.'
            )
            loaded_state_dict.pop(key)

    # Load pretrained weights
    model_state_dict.update(loaded_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)

    return model, data_scaler


def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]

    return Adam(params)


def build_lr_scheduler(
    optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None
) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )


def create_logger(
    name: str, save_dir: str = None, quiet: bool = False
) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, "verbose.log"))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, "quiet.log"))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """

    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """

        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = (
                logging.getLogger(logger_name).info
                if logger_name is not None
                else print
            )
            info(f"Elapsed time = {delta}")

            return result

        return wrap

    return timeit_decorator


def make_noise(batch_size, shape, device, volatile=False):
    tensor = torch.randn(batch_size, shape)
    noise = Variable(tensor, volatile)
    noise = noise.to(device, dtype=torch.float32)
    return noise


def pearson_mean(data1, data2):
    sum_pearson_1 = 0
    sum_pearson_2 = 0
    for i in range(data1.shape[0]):
        pearsonr_ = pearsonr(data1[i], data2[i])
        sum_pearson_1 += pearsonr_[0]
        sum_pearson_2 += pearsonr_[1]
    return sum_pearson_1 / data1.shape[0], sum_pearson_2 / data1.shape[0]


def r2_mean(data1, data2):
    sum_r2_1 = 0
    for i in range(data1.shape[0]):
        r2_score_ = r2_score(data1[i], data2[i])
        sum_r2_1 += r2_score_
    return sum_r2_1 / data1.shape[0]
