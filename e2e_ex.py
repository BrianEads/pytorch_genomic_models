import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# --- 1. Model Definition ---
class SimpleGenomicClassifier(nn.Module):
    def __init__(self, num_tokens=5, embedding_dim=32, seq_len=101):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=8)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened output after convolution and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, seq_len, dtype=torch.long)
            dummy_embedded = self.embedding(dummy_input).permute(0, 2, 1)
            dummy_conv = self.conv1d(dummy_embedded)
            pool_kernel_size = dummy_conv.shape[2]  # Global Max Pooling
            dummy_pool = nn.MaxPool1d(kernel_size=pool_kernel_size)(dummy_conv)
            dummy_flattened = self.flatten(dummy_pool)
            flattened_size = dummy_flattened.shape[1]

        self.linear = nn.Linear(flattened_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) # (B, L) -> (B, L, D)
        x = x.permute(0, 2, 1) # (B, L, D) -> (B, D, L) for Conv1d
        x = self.conv1d(x) # (B, D, L) -> (B, C_out, L_out)
        x = self.relu(x)
        
        # Global Max Pooling
        pool_kernel_size = x.shape[2]
        x = nn.MaxPool1d(kernel_size=pool_kernel_size)(x)
        
        x = self.flatten(x) # (B, C_out, 1) -> (B, C_out)
        x = self.linear(x) # (B, C_out) -> (B, 1)
        x = self.sigmoid(x)
        return x

# --- 2. Data Preparation ---
def generate_synthetic_data(num_samples=1000, seq_len=101):
    # Vocab: 0=PAD, 1=A, 2=C, 3=G, 4=T
    sequences = np.random.randint(1, 5, size=(num_samples, seq_len))
    # Labels: 1 if '123' (ACG) is in the sequence, 0 otherwise
    labels = np.array([1 if any(np.array_equal(sequences[i, j:j+3], [1, 2, 3]) for j in range(seq_len - 2)) else 0 for i in range(num_samples)])
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# --- 3. Training Loop ---
if __name__ == '__main__':
    # Hyperparameters
    SEQ_LEN = 101
    NUM_SAMPLES = 2000
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001

    # Generate data and create DataLoader
    X, y = generate_synthetic_data(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate model, loss function, and optimizer
    model = SimpleGenomicClassifier(seq_len=SEQ_LEN)
    criterion = nn.BCELoss() # For binary classification with sigmoid output
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    print("Training finished.")
