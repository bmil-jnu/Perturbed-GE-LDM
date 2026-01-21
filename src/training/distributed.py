"""
Distributed training utilities.
"""

import os
import datetime
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedManager:
    """
    Manager for distributed training setup and teardown.
    
    Handles:
    - Process group initialization
    - Device assignment
    - DDP model wrapping
    - Cleanup
    
    Example:
        >>> dist_manager = DistributedManager(world_size=4)
        >>> dist_manager.setup(rank=0)
        >>> model = dist_manager.wrap_model(model)
        >>> # ... training ...
        >>> dist_manager.cleanup()
    """
    
    def __init__(
        self,
        world_size: int = 1,
        backend: str = "nccl",
        master_addr: str = "localhost",
        master_port: str = "12356",
        timeout_days: int = 2,
    ):
        """
        Initialize DistributedManager.
        
        Args:
            world_size: Total number of processes
            backend: Distribution backend (nccl, gloo, etc.)
            master_addr: Address of master node
            master_port: Port for communication
            timeout_days: Timeout in days
        """
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.timeout = datetime.timedelta(days=timeout_days)
        
        self.rank = 0
        self.device = None
        self._initialized = False
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    def setup(self, rank: int) -> None:
        """
        Set up distributed training for this process.
        
        Args:
            rank: Rank of this process
        """
        self.rank = rank
        
        if not self.is_distributed:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return
        
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=self.world_size,
            timeout=self.timeout
        )
        
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)
        self._initialized = True
        
        print(f"Process {rank}/{self.world_size} initialized on {self.device}")
    
    def wrap_model(
        self,
        model: torch.nn.Module,
        find_unused_parameters: bool = False,
    ) -> torch.nn.Module:
        """
        Wrap model with DistributedDataParallel.
        
        Args:
            model: Model to wrap
            find_unused_parameters: Whether to find unused parameters
            
        Returns:
            Wrapped model (or original if not distributed)
        """
        if not self.is_distributed:
            return model.to(self.device)
        
        model = model.to(self.rank)
        return DDP(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=find_unused_parameters
        )
    
    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self._initialized:
            dist.barrier()
            dist.destroy_process_group()
            self._initialized = False
            print(f"Process {self.rank} cleaned up")
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self._initialized:
            dist.barrier()
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """
        Broadcast tensor from source to all processes.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
            
        Returns:
            Broadcasted tensor
        """
        if self._initialized:
            dist.broadcast(tensor, src=src)
        return tensor
    
    def all_gather_scalar(self, value: float) -> list:
        """
        Gather scalar values from all processes.
        
        Args:
            value: Scalar value to gather
            
        Returns:
            List of values from all processes
        """
        if not self._initialized:
            return [value]
        
        tensor = torch.tensor([value], device=self.device)
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return [t.item() for t in gathered]
    
    def reduce_mean(self, value: float) -> float:
        """
        Reduce scalar value by averaging across all processes.
        
        Args:
            value: Scalar value
            
        Returns:
            Averaged value
        """
        values = self.all_gather_scalar(value)
        return sum(values) / len(values)


def setup_distributed(rank: int, world_size: int) -> None:
    """
    Simple function to set up distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(days=2)
    )
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
