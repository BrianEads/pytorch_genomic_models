import torch
import torch.optim as optim
import torch.nn as nn

# --- 1. Setup ---
# Assume we have a model, some input data, and some true labels
model = nn.Linear(10, 1) # A simple example model
input_data = torch.randn(64, 10)
true_labels = torch.randn(64, 1)
loss_function = nn.MSELoss() # Mean Squared Error loss

# --- 2. Create the Optimizer ---
# We pass the model's parameters to the optimizer.
# 'lr' is the learning rate.
# 'weight_decay' is a regularization term (more on this next).
# AdamW is a great default choice.
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# --- 3. A Single Training Step ---
# a. Clear old gradients
optimizer.zero_grad()

# b. Forward pass: get model predictions
predictions = model(input_data)

# c. Compute the loss
loss = loss_function(predictions, true_labels)

# d. Backward pass: compute gradients
loss.backward()

# e. Update weights: take a step based on the gradients
optimizer.step()

# --- 4. Observe ---
print("Loss after one step:", loss.item())
