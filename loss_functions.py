import torch
import torch.nn as nn

# --- 1. Binary Classification Example ---
# Task: Predict if 4 sequences are binding sites (1) or not (0).
# Model outputs raw scores (logits). Positive scores -> class 1, Negative -> class 0.
logits = torch.tensor([-2.5, 4.1, -0.5, 1.1]) # Raw output from a model for a batch of 4
true_labels = torch.tensor([0.0, 1.0, 0.0, 1.0]) # The ground truth (as floats)

# BCEWithLogitsLoss is best for binary tasks.
loss_fn_bce = nn.BCEWithLogitsLoss()
binary_loss = loss_fn_bce(logits, true_labels)
print(f"Binary Cross-Entropy Loss: {binary_loss.item():.4f}")


# --- 2. Multi-Class Classification Example ---
# Task: Classify 2 residues into one of 3 classes (helix, sheet, coil).
# Model outputs a score for each class.
logits_multi = torch.tensor([
    [3.2, -1.0, 0.5],  # Logits for residue 1 (class 0 is highest)
    [-0.8, 2.5, 0.1]   # Logits for residue 2 (class 1 is highest)
])
true_labels_multi = torch.tensor([...](asc_slot://start-slot-228)) # The ground truth (as integers)

# CrossEntropyLoss is best for multi-class tasks.
loss_fn_ce = nn.CrossEntropyLoss()
multiclass_loss = loss_fn_ce(logits_multi, true_labels_multi)
print(f"Multi-Class Cross-Entropy Loss: {multiclass_loss.item():.4f}")
