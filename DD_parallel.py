# This is a conceptual script, e.g., `my_ddp_script.py`
# You would run it from the command line like:
# torchrun --nproc_per_node=4 my_ddp_script.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

# Dummy class for the example to be runnable
class MyDataset(Dataset):
    def __len__(self): return 1000
    def __getitem__(self, idx): return torch.randn(10), torch.randn(1)

def main_worker(rank, world_size):
    """The main training function for each process."""
    setup(rank, world_size)
    
    # 1. Wrap the model with DDP
    # The model is moved to the GPU corresponding to the process's rank.
    model = nn.Linear(10, 1).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 2. Use DistributedSampler for the DataLoader
    # This ensures each process gets a different slice of the data.
    dataset = MyDataset() 
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # Note: shuffle=False because the sampler handles shuffling.
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, shuffle=False) 

    # ... standard training loop using `ddp_model` ...
    # The gradient synchronization is handled automatically by DDP during loss.backward().

    # Only save the model on the main process (rank 0) to avoid conflicts.
    if rank == 0:
        torch.save(ddp_model.state_dict(), "my_model.pt")

    cleanup()

if __name__ == '__main__':
    # This part is handled by torchrun, which sets environment variables.
    # world_size = number of GPUs
    # rank = the ID of the current GPU (0, 1, 2, ...)
    # For a conceptual example, we'll skip the process spawning part.
    print("To run a DDP script, use 'torchrun'.")
    print("This example shows the key code modifications needed within the script.")
