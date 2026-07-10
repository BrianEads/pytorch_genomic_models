import torch
import torch.nn as nn
import math

# --- Positional Encoding Module ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- Transformer Example ---
# 1. Setup
seq_length = 100
embedding_dim = 16
num_heads = 4
num_layers = 2 # Stack 2 transformer layers
batch_size = 1

# Assume we have an embedded sequence
# Shape: (seq_length, batch_size, embedding_dim)
embedded_input = torch.randn(seq_length, batch_size, embedding_dim)

# 2. Add Positional Encodings
pos_encoder = PositionalEncoding(d_model=embedding_dim)
positioned_input = pos_encoder(embedded_input)

# 3. Create the Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embedding_dim, 
    nhead=num_heads,
    batch_first=False # Our input is (Seq, Batch, Dim)
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# 4. Apply the Transformer
output = transformer_encoder(positioned_input)

# 5. Observe the Output
print("Shape of Transformer output:", output.shape)
