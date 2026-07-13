# K-mer Pre-training for Genomic Foundation Models

In the pre-training of genomic foundational models like DNABERT, the k-mer tokenization strategy is a crucial step that transforms raw DNA sequences into a format that can be processed by language models. This approach, inspired by n-grams in natural language processing, involves breaking down DNA sequences into smaller, overlapping subsequences of a fixed length 'k'.

## K-mer Tokenization Strategy: Rationale and Methodology
The core idea behind k-mer tokenization is to represent a DNA sequence as a series of these fixed-length "words" rather than individual nucleotides. For instance, the DNA sequence 'ATGGCT' would be tokenized into a sequence of four 3-mers: {ATG, TGG, GGC, GCT}. 
 This is typically done using a sliding window of size 'k' with a stride of 1, creating overlapping k-mers.

### Sliding-window example (k=4)

```text
Sequence:  A T G C A G T T A C G A
           |-------|               k=4, step=1
               |-------|
                   |-------|
                       |-------|
                           |-------|
                               |-------|
                                   |-------|
                                       |-------|
                                           |-------|

Tokens:  ATGC → id:42
         TGCA → id:17
         GCAG → id:83
         CAGT → id:29
         ...
```


Rationale for Overlapping k-mers:

Using overlapping k-mers instead of single nucleotides allows the model to capture richer contextual information for each base by concatenating it with its neighbors. 
 This is biologically significant as many functional elements in DNA, such as transcription factor binding sites and splice sites, are defined by short sequence motifs. Single nucleotide tokenization would require the model to learn these local patterns from scratch, whereas k-mer tokenization provides these patterns as direct inputs. Overlapping k-mers, in particular, ensure a dense representation of the local context. 

Advantages of k-mer Tokenization:
Capturing Biological Context: Overlapping k-mers can preserve local sequence context and represent important biological motifs within the genomic sequences. 
 This helps the model to learn the "language" of DNA more effectively. 

Computational Efficiency: Compared to one-hot encoding of individual nucleotides, k-mer tokenization can be more efficient.

Training-Free: Unlike vocabulary-learning methods like Byte-Pair Encoding (BPE), k-mer approaches do not require a separate training phase to build the vocabulary. 

Disadvantages of k-mer Tokenization:
Information Leakage: With overlapping k-mers, adjacent tokens share a significant number of nucleotides. This can lead to "information leakage" during masked language modeling, where the model can infer a masked token from its immediate neighbors without learning deeper contextual relationships. 

Fixed-Length Limitation: K-mers have a fixed length, which can be a drawback when dealing with biological motifs of varying lengths. 

Vocabulary Size: The size of the vocabulary grows exponentially with the value of 'k' (4^k). This can lead to a very large vocabulary for larger 'k' values, increasing the model's memory footprint. 

Data Sparsity and Imbalance: The distribution of k-mers in a genome is often highly imbalanced, with some k-mers appearing very frequently and others being extremely rare. This can make it challenging for the model to learn meaningful patterns from rare sequences. 

Impact of Choosing 'k':
The choice of 'k' involves a trade-off between capturing sufficient context and managing computational resources.

Smaller 'k' (e.g., 3, 4):

Results in a smaller, more manageable vocabulary (4^3=64, 4^4=256). 

Provides better sequence coverage, especially with non-overlapping tokenization strategies. 

May not capture the complexity of longer biological motifs. 

Larger 'k' (e.g., 5, 6, 8):

Creates a much larger vocabulary (4^6 = 4,096, 4^8 = 65,541). 

Can encapsulate more specific and longer biological motifs.

Can lead to increased data sparsity and computational demands. 

In practice, DNABERT was trained with different models for k=3, 4, 5, and 6 to evaluate their performance on various downstream tasks. 

## The Data Processing Pipeline
The process of converting a raw DNA sequence into a format suitable for a model like DNABERT involves several steps:

Vocabulary Creation:
A vocabulary of all possible k-mers for a given 'k' is generated. For DNA, with its four bases (A, C, G, T), this results in 4^k possible k-mers. In addition to these, five special tokens are added to the vocabulary 
 :

[CLS]: A classification token added to the beginning of each sequence. Its embedding is often used for sequence-level classification tasks. 

[SEP]: A separator token to denote the end of a sequence. 

[MASK]: A token used to replace a k-mer during masked language modeling. 

[PAD]: A padding token used to make all sequences in a batch have the same length. 

[UNK]: An "unknown" token for any k-mer not in the vocabulary, though this is rare with a complete k-mer vocabulary. 

The total vocabulary size is therefore 4^k + 5. 

Sequence Tokenization:
A raw DNA sequence is converted into a sequence of k-mers. For example, using a 6-mer tokenizer, a sliding window of length 6 moves across the sequence one nucleotide at a time, generating a list of overlapping 6-mers. 

Adding Special Tokens and Conversion to IDs:
The [CLS] token is prepended to the sequence of k-mers, and the [SEP] token is appended. 
 Each token (k-mer and special tokens) in the sequence is then mapped to its corresponding integer ID from the created vocabulary.

## Masked Language Modeling (MLM) and the 15% Masking Strategy
DNABERT is pre-trained using a Masked Language Modeling (MLM) objective, similar to BERT. 
 This self-supervised learning task involves masking some of the tokens in the input sequence and training the model to predict the original tokens. 

The standard strategy is to randomly select 15% of the input tokens for potential replacement. 
 Of these selected tokens:

80% are replaced with the [MASK] token.

10% are replaced with a random token from the vocabulary.

10% are left unchanged.

The model is then trained to predict the original tokens at the masked positions by minimizing the cross-entropy loss between the model's predictions and the true tokens. 
 This forces the model to learn a deep, bidirectional representation of the DNA "language."

## Python Code Example
Here is a Python code example demonstrating the entire process, from k-mer tokenization and masking to preparing the data for a model. This example uses concepts similar to those in the Hugging Face transformers library.

```python
import torch
import random
from itertools import product

# --- 1. K-mer Tokenization and Vocabulary ---

def seq_to_kmer(seq, k):
    """Converts a DNA sequence into a list of overlapping k-mers."""
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def build_kmer_vocabulary(k, special_tokens):
    """Creates a vocabulary for k-mers and special tokens."""
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    
    # Create token to id mapping
    vocab = {token: i for i, token in enumerate(special_tokens + kmers)}
    # Create id to token mapping
    id_to_token = {i: token for token, i in vocab.items()}
    
    return vocab, id_to_token

# --- 2. Data Processing Pipeline ---

def tokenize_sequence(sequence, k, vocab):
    """Converts a raw DNA sequence to a list of token IDs."""
    kmer_list = seq_to_kmer(sequence, k)
    
    # Add special tokens
    tokenized_kmer_sequence = ['[CLS]'] + kmer_list + ['[SEP]']
    
    # Convert tokens to IDs
    input_ids = [vocab.get(token, vocab['[UNK]']) for token in tokenized_kmer_sequence]
    
    return input_ids

# --- 3. Masked Language Modeling (MLM) Implementation ---

def mask_tokens(inputs, vocab, mask_prob=0.15):
    """
    Prepares masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    
    # Probability matrix for masking
    prob_matrix = torch.full(labels.shape, mask_prob)
    
    # Determine which tokens to mask
    special_tokens_mask = [
        val in [vocab['[CLS]'], vocab['[SEP]'], vocab['[PAD]']] for val in labels.tolist()
    ]
    prob_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with [MASK] token
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = vocab['[MASK]']

    # 10% of the time, we replace masked input tokens with a random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(vocab), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The remaining 10% of the time (10% of 15% overall) we keep the original token
    # This is implicitly handled as we do nothing to these tokens.
    
    return inputs, labels

# --- Main Example ---

# Configuration
K = 6  # k-mer size
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
DNA_SEQUENCE = "ACGTAGCTAGCTAGCTACGATCGATCGATCGATACGATCGATCG"

# 1. Build Vocabulary
vocab, id_to_token = build_kmer_vocabulary(K, SPECIAL_TOKENS)
print(f"Vocabulary size: {len(vocab)}")
print("-" * 30)

# 2. Tokenize the raw DNA sequence
input_ids = tokenize_sequence(DNA_SEQUENCE, K, vocab)
print(f"Original DNA Sequence: {DNA_SEQUENCE}")
print(f"Tokenized k-mers (first 5): {seq_to_kmer(DNA_SEQUENCE, K)[:5]}")
print(f"Input IDs (first 10): {input_ids[:10]}")
print("-" * 30)

# 3. Apply Masking
inputs_tensor = torch.tensor(input_ids)
masked_inputs, labels = mask_tokens(inputs_tensor.clone(), vocab)

print("Original Input IDs:\n", inputs_tensor)
print("\nMasked Input IDs (some tokens are replaced):\n", masked_inputs)
print("\nLabels (only masked tokens have values, others are -100):\n", labels)
print("-" * 30)


# --- 4. Model Output Configuration for Prediction ---

# This is a conceptual example of how a model would be used.
# It requires a pre-trained model from a library like transformers.

# from transformers import BertForMaskedLM, BertConfig

# # Assuming we have a pre-trained DNABERT model
# config = BertConfig(vocab_size=len(vocab), num_hidden_layers=12, num_attention_heads=12)
# model = BertForMaskedLM(config)

# # Move tensors to the appropriate device (e.g., GPU)
# # masked_inputs = masked_inputs.to(device)
# # labels = labels.to(device)

# # During training, you would pass both inputs and labels to the model
# # The model computes the loss internally
# # outputs = model(input_ids=masked_inputs.unsqueeze(0), labels=labels.unsqueeze(0))
# # loss = outputs.loss

# print("Conceptual Model Training:")
# print("The 'masked_inputs' tensor would be the input to the model.")
# print("The 'labels' tensor would be used to calculate the cross-entropy loss against the model's predictions for the masked positions.")
# print("The model's goal is to predict the original token IDs where the labels are not -100.")

```
This comprehensive process of k-mer tokenization and masked language modeling allows genomic foundational models to learn the complex patterns and "grammar" of DNA, enabling powerful performance on a wide range of downstream genomic tasks. While effective, the limitations of k-mer tokenization have led to the development of other methods like Byte-Pair Encoding (BPE) in newer models such as DNABERT-2.
