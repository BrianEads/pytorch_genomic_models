import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# This code requires a CUDA-enabled GPU.

# --- 1. Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Mixed precision requires a CUDA GPU. Skipping example.")
else:
    model = nn.Linear(10, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Create a GradScaler once at the beginning of training.
    scaler = GradScaler()

    # Dummy data
    input_data = torch.randn(64, 10, device=device)
    true_labels = torch.randn(64, 1, device=device)

    # --- 2. A Single Mixed-Precision Training Step ---
    optimizer.zero_grad()

    # Wrap the forward pass with autocast.
    # Operations inside this block will run in lower precision where possible.
    with autocast(device_type='cuda', dtype=torch.float16):
        predictions = model(input_data)
        loss = loss_fn(predictions, true_labels)

    # Scale the loss and call backward() on the scaled loss.
    scaler.scale(loss).backward()

    # scaler.step() first unscales the gradients and then calls optimizer.step().
    scaler.step(optimizer)

    # Update the scale for the next iteration.
    scaler.update()

    print("Training step completed with mixed precision.")
    print("Loss:", loss.item())
