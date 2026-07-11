import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# --- 1. Create a dummy dataset with variable length sequences ---
class GenomicDataset(Dataset):
    def __init__(self):
        self.data = [
            torch.tensor([...](asc_slot://start-slot-222)),          # len 4
            torch.tensor([...](asc_slot://start-slot-224)),             # len 3
            torch.tensor([...](asc_slot://start-slot-226))    # len 6
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- 2. Define the custom collate function ---
# This function will be called by the DataLoader for each batch.
def pad_collate_fn(batch):
    # 'batch' is a list of tensors (our sequences)
    # We define our padding token's index (e.g., 4 for 'N')
    padding_value = 4
    
    # pad_sequence stacks the tensors and pads them to the longest sequence in the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
    return padded_batch

# --- 3. Use it with a DataLoader ---
dataset = GenomicDataset()
# batch_size=3 will grab all our data in one go
data_loader = DataLoader(dataset, batch_size=3, collate_fn=pad_collate_fn)

# --- 4. Observe the Output ---
# Get one batch from the loader
padded_sequences = next(iter(data_loader))

print("Padded batch of sequences:\n", padded_sequences)
print("Shape of the batch:", padded_sequences.shape)
