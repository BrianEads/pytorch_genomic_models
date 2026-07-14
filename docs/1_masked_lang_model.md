# Masked Language Modeling for Genomic Sequences

The application of deep learning to unravel the complexities of genomic data has led to significant breakthroughs. Foundational biological models, particularly those leveraging the Transformer architecture, have become instrumental in understanding the language of life. We seek to leverage Deep Learning Genomic models for crop science projects like trait discovery, target/MoA elucidation, and editing. This project starts from core concepts underpinning these models, including Masked Language Modeling for pre-training, the critical role of positional encodings, and provides practical PyTorch implementations for genomic sequence analysis.

## Masked Language Modeling (MLM) for Genomic Sequences
Masked Language Modeling (MLM) is a self-supervised pre-training objective that has been successfully adapted from natural language processing (NLP) to genomics. In essence, MLM trains a model to predict missing or "masked" portions of a sequence based on the surrounding context. For genomic sequences, this means randomly hiding nucleotides (A, C, G, T) and tasking the model with their reconstruction. 

Application to Genomic Sequences:

When applied to a huge number of DNA or RNA sequences, MLM enables a model to learn the intricate "grammar" and syntax of the genome. By predicting masked nucleotides, the model is compelled to understand several key features that are useful for our purposes:
1. Contextual Dependencies: It learns the relationships between different parts of a genomic sequence, recognizing that the identity of a nucleotide is often influenced by its neighbors. 
2. Motif Discovery: The model can implicitly learn to identify functionally important sequence patterns, known as motifs (e.g., transcription factor binding sites), as these are crucial for accurately predicting masked elements. 
3. Long-Range Interactions: Transformer-based models with MLM can capture dependencies between distant elements in a sequence, which is vital for understanding complex regulatory mechanisms in DNA. 

Foundational models like DNABERT and the Nucleotide Transformer are pre-trained on massive genomic datasets using MLM. 
This pre-training endows them with a rich, contextual understanding of genomic sequences, which can then be fine-tuned for a variety of specific downstream tasks like promoter prediction, splice-site identification, and transcription factor binding site prediction. 

## The Role and Necessity of Positional Encodings
Transformer models, by their design, do not inherently process sequential data in order. The self-attention mechanism, which is at the core of the Transformer, treats the input as an unordered set of tokens. 
This "permutation invariance" means that if you were to shuffle the nucleotides in a DNA sequence, the model would produce the same output, which is clearly undesirable for biological sequence analysis where order is paramount. 

To address this, positional encodings are used to inject information about the relative or absolute position of each token (nucleotide) in the sequence. This is typically done by adding a unique vector, the positional encoding, to each token's embedding at the input layer of the model. 

For biological sequences, this is crucial because the function of a nucleotide is inextricably linked to its position within a gene, a regulatory element, or the genome at large. Without positional information, a model would be unable to distinguish between a functional motif and a random assortment of the same nucleotides. Sinusoidal positional encodings, as proposed in the original Transformer paper, are a common choice as they can generalize to sequences of different lengths.
Custom relative positional encodings have also been explored to better capture the distance-dependent effects of regulatory elements like enhancers. 

## PyTorch Example: Sinusoidal Positional Encodings
Here is a PyTorch implementation of sinusoidal positional encodings. This code generates a matrix of positional encodings that can be added to the token embeddings of a genomic sequence.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            dropout (float): The dropout probability.
            max_len (int): The maximum length of the input sequences.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of token embeddings.
                              Shape: (sequence_length, batch_size, d_model)
        
        Returns:
            torch.Tensor: The input tensor with added positional encodings.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Example Usage:
d_model = 64  # Embedding dimension
max_sequence_length = 200
batch_size = 10

# Create a dummy token embedding tensor
token_embeddings = torch.randn(max_sequence_length, batch_size, d_model)

# Instantiate and apply positional encoding
pos_encoder = PositionalEncoding(d_model)
encoded_embeddings = pos_encoder(token_embeddings)

print("Shape of original embeddings:", token_embeddings.shape)
print("Shape of embeddings with positional encoding:", encoded_embeddings.shape)
```
## End-to-End PyTorch Model for Genomic Sequence Classification
For tasks like transcription factor binding site (TFBS) prediction, a common approach is to use a Convolutional Neural Network (CNN) to detect local motifs, followed by other layers to process this information. 
 Below is a complete, end-to-end PyTorch example of a simple model for a binary classification task like TFBS prediction. This model combines an embedding layer, a 1D convolutional layer, a pooling layer, and a final linear layer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 1. Data Preparation ---

# Helper function to one-hot encode a DNA sequence
def one_hot_encode(seq, max_len=200):
    """
    One-hot encodes a DNA sequence.
    'N' is encoded as all zeros.
    """
    seq = seq.upper()
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encoded_seq = np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq])
    
    # Pad or truncate the sequence
    if len(encoded_seq) > max_len:
        encoded_seq = encoded_seq[:max_len]
    elif len(encoded_seq) < max_len:
        padding = np.zeros((max_len - len(encoded_seq), 4))
        encoded_seq = np.vstack([encoded_seq, padding])
        
    return encoded_seq.T  # Transpose to get (channels, sequence_length) which is (4, 200)

# Custom PyTorch Dataset
class GenomicDataset(Dataset):
    def __init__(self, sequences, labels, max_len=200):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # One-hot encode the sequence
        encoded_sequence = one_hot_encode(sequence, self.max_len)
        
        return torch.tensor(encoded_sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# --- 2. Model Definition ---

class SimpleGenomicClassifier(nn.Module):
    def __init__(self):
        super(SimpleGenomicClassifier, self).__init__()
        # PyTorch's Conv1d expects input of shape (batch_size, in_channels, sequence_length)
        # For our one-hot encoded DNA, in_channels is 4 (A, C, G, T)
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=8, stride=1, padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        # Flatten the output for the linear layer
        self.flatten = nn.Flatten()
        
        # The input features to the linear layer depend on the output of the pooling layer
        # Seq length after conv: 200 (due to 'same' padding)
        # Seq length after pool: 200 / 4 = 50
        # Flattened size: 16 channels * 50 = 800
        self.fc1 = nn.Linear(16 * 50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

# --- 3. Training Loop ---

if __name__ == '__main__':
    # Generate some dummy data for demonstration
    num_samples = 100
    sequence_length = 200
    
    # Positive samples with a motif
    motif = "GATTACA"
    positive_sequences = ["".join(np.random.choice(list("ACGT"), sequence_length - len(motif))) + motif for _ in range(num_samples // 2)]
    positive_labels = [1] * (num_samples // 2)
    
    # Negative samples without the motif
    negative_sequences = ["".join(np.random.choice(list("ACGT"), sequence_length)) for _ in range(num_samples // 2)]
    negative_labels = [0] * (num_samples // 2)
    
    all_sequences = positive_sequences + negative_sequences
    all_labels = positive_labels + negative_labels
    
    # Create dataset and dataloader
    dataset = GenomicDataset(all_sequences, all_labels, max_len=sequence_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = SimpleGenomicClassifier()
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training process
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequences, labels in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training finished.")

    # --- 4. Prediction on a new sequence ---
    model.eval()
    new_sequence = "".join(np.random.choice(list("ACGT"), sequence_length - 10)) + "GATTACA" + "NN"
    encoded_new_seq = torch.tensor(one_hot_encode(new_sequence, max_len=sequence_length), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prediction = model(encoded_new_seq)
        print(f"\nPrediction for new sequence containing the motif: {prediction.item():.4f}")
        if prediction.item() > 0.5:
            print("Predicted class: Binding Site")
        else:
            print("Predicted class: Not a Binding Site")

```
This comprehensive example illustrates the practical application of deep learning for a fundamental task in genomics, from data representation to model training and prediction.

## Sequence position diagram

```text
pos:  1   2   3   4   5   6   7   8   9  10  11  12
seq:  A   T   G   C   A   G   T   T   A   C   G   A
```
