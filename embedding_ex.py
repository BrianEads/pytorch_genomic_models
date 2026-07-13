import torch
import torch.nn as nn

# --- 1. Setup ---
# Vocabulary: 0=A, 1=C, 2=G, 3=T, 4=N (padding/unknown)
vocab_size = 5
# Each nucleotide will be represented by a vector of size 16
embedding_dim = 16

# Create the embedding layer
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# --- 2. Example DNA Sequence ---
# Let's represent the sequence "ACGT" as integer indices
# This is our model's input
dna_sequence_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

# --- 3. Apply Embedding ---
# Pass the indices through the embedding layer
embedded_sequence = embedding_layer(dna_sequence_indices)

# --- 4. Observe the Output ---
# The output is a tensor where each integer index has been replaced
# by a dense vector of size 'embedding_dim'.
# Shape: (sequence_length, embedding_dim)
print("Shape of embedded sequence:", embedded_sequence.shape)
print("Embedded 'A':\n", embedded_sequence[0])
