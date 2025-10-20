from cross_validate import cross_validate
from run_training import run_training
import torch
import torch.multiprocessing as mp
from datetime import datetime


def main():

    start_time = datetime.now()

    world_size = torch.cuda.device_count()
    mp.spawn(
        cross_validate,
        args=(world_size, run_training),
        nprocs=world_size,
    )

    end_time = datetime.now()
    during_time = (end_time - start_time).seconds / 60

    print(f"start time: {start_time}, endtime: {end_time}, time: {during_time} mins")


if __name__ == "__main__":
    main()
