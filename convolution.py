import torch
import torch.nn as nn

# --- 1. Setup (Continuing from Embedding) ---
batch_size = 1
seq_length = 100
embedding_dim = 16 # This is our number of "input channels"

# A random embedded sequence
# Shape: (batch_size, seq_length, embedding_dim)
embedded_sequence = torch.randn(batch_size, seq_length, embedding_dim)

# PyTorch convolutions expect (Batch, Channels, Length), so we permute the dimensions
embedded_sequence = embedded_sequence.permute(0, 2, 1) # Shape becomes (1, 16, 100)

# --- 2. Create the Convolutional Layer ---
# in_channels: Must match the embedding dimension
# out_channels: The number of different motifs we want to learn (e.g., 32 different scanners)
# kernel_size: The width of the scanner (e.g., a motif of length 8)
conv_layer = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=8)

# --- 3. Apply Convolution ---
conv_output = conv_layer(embedded_sequence)

# --- 4. Observe the Output ---
# The output shape will be (batch_size, num_motifs, new_length)
# The length is slightly smaller due to the kernel size.
print("Shape after 1D convolution:", conv_output.shape)
