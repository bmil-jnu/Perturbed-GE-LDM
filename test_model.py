from cross_validate import cross_validate
from run_test import run_test
import torch
import torch.multiprocessing as mp
from datetime import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    start_time = datetime.now()
    world_size = torch.cuda.device_count()

    if world_size == 1:
        print("Single GPU detected.")
        cross_validate(rank=0, world_size=world_size, train_func=run_test, parallel=False)
    else:
        print(f"{world_size} GPUs detected.")
        parallel = True
        mp.spawn(
        cross_validate,
        args=(world_size, run_test, parallel),
        nprocs=world_size,
    )

    end_time = datetime.now()
    during_time = (end_time - start_time).seconds / 60

    print(f"start time: {start_time}, endtime: {end_time}, time: {during_time} mins")


if __name__ == "__main__":
    main()
