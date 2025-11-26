import os
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'

from cross_validate import cross_validate
from run_training import run_training
import torch
import torch.multiprocessing as mp
from datetime import datetime


def main():
    start_time = datetime.now()
    world_size = torch.cuda.device_count()

    if world_size == 1:
        cross_validate(rank=0, world_size=world_size, train_func=run_training, parallel=False)
    else:
        parallel = True
        mp.spawn(
        cross_validate,
        args=(world_size, run_training, parallel),
        nprocs=world_size,
    )

    end_time = datetime.now()
    during_time = (end_time - start_time).seconds / 60

    print(f"start time: {start_time}, endtime: {end_time}, time: {during_time} mins")


if __name__ == "__main__":
    main()
