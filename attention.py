import torch
import torch.nn as nn

# --- 1. Setup ---
batch_size = 1
seq_length = 100
embedding_dim = 16 # The model's internal dimension

# A random embedded sequence from a previous layer
# Shape: (seq_length, batch_size, embedding_dim) - Attention layers prefer this format
input_sequence = torch.randn(seq_length, batch_size, embedding_dim)

# --- 2. Create the Attention Layer ---
# embed_dim: The model's dimension
# num_heads: How many attention mechanisms to run in parallel. Must be a divisor of embed_dim.
attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)

# --- 3. Apply Attention ---
# In self-attention, the query, key, and value are all the same input sequence.
# The model attends to different parts of itself.
# We set need_weights=True to get the attention matrix for interpretation.
attn_output, attn_weights = attention_layer(input_sequence, input_sequence, input_sequence,
                                            need_weights=True)

# --- 4. Observe the Output ---
# attn_output has the same shape as the input, but each position's vector
# is now a context-aware representation.
print("Shape of attention output:", attn_output.shape)

# attn_weights show the learned importance scores.
# Shape: (batch_size, seq_length, seq_length)
# attn_weights[0, i, j] is how much position 'i' paid attention to position 'j'.
print("Shape of attention weights:", attn_weights.shape)
