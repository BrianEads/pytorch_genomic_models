import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Dropout Example ---
# Dropout is typically applied after activation functions or between linear layers.
# p=0.5 means each neuron has a 50% chance of being zeroed out during training.
# Note: Dropout is automatically disabled during evaluation (model.eval()).
dropout_layer = nn.Dropout(p=0.5)
activations = torch.randn(20, 16) # Example activations from a previous layer
print("Original activations (first 5):\n", activations[0, :5])
dropped_out_activations = dropout_layer(activations)
print("Activations after dropout (first 5):\n", dropped_out_activations[0, :5])


# --- 2. Weight Decay Example ---
# Weight decay is specified when you create the optimizer.
model = nn.Linear(10, 1)

# AdamW correctly decouples weight decay from the gradient update, making it a preferred choice.
optimizer_with_wd = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print("\nOptimizer with weight decay:", optimizer_with_wd)
