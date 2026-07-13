import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Minimal self-attention setup so this script runs standalone.
batch_size = 1
seq_length = 100
embedding_dim = 16
input_sequence = torch.randn(seq_length, batch_size, embedding_dim)
attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
_, attn_weights = attention_layer(
    input_sequence, input_sequence, input_sequence, need_weights=True
)

# Visualize attention weights for the first sequence in the batch.
attention_matrix = attn_weights[0].detach().cpu().numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(attention_matrix, cmap="viridis")
plt.title("Attention Heatmap")
plt.xlabel("Key Positions (Attended To)")
plt.ylabel("Query Positions (Attending From)")
plt.show()
