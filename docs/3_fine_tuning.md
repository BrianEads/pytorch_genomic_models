# Fine-Tuning Pre-trained Genomic Foundation Models

An in-depth look at fine-tuning pre-trained genomic foundational models reveals a powerful technique for adapting these large-scale models to specific biological questions. This process, coupled with methods to interpret the learned features, allows researchers to not only make accurate predictions but also to gain biological insights.

## Fine-Tuning Pre-trained Genomic Foundational Models
Fine-tuning adapts a model that has been pre-trained on a vast amount of general genomic data to a more specialized task, such as predicting transcription factor binding sites. 
 This approach is advantageous because the pre-trained model has already learned a rich, generalizable representation of genomic sequences, which can then be leveraged for a new task with a much smaller, task-specific labeled dataset. 

### Architectural Modifications
The primary architectural modification required for fine-tuning a pre-trained model for a new classification task is the replacement or addition of a task-specific "head". 
 Foundational models are often pre-trained on self-supervised tasks, and their final layers are not suited for a specific downstream task.

For a classification task like predicting transcription factor binding sites, the process involves:

Loading the Pre-trained Model: The core of the new model is the pre-trained genomic foundational model.

Removing the Original Head: The final layer or layers of the pre-trained model, which were used for the pre-training objective, are removed.

Adding a New Classification Head: A new set of layers, typically one or more fully connected (linear) layers, is added on top of the pre-trained base. 
 The final layer of this new head will have an output size equal to the number of classes in the new task (e.g., two for a binary classification of "binding" vs. "no binding"). A softmax or sigmoid activation function is often used in the final layer to produce probabilities.

### Training Strategies
There are several strategies for training the newly constructed model, each with different implications for computational cost and performance:

Feature Extraction (Freezing the Base): In this approach, the weights of the pre-trained model (the "backbone") are frozen, meaning they are not updated during training. 
 Only the weights of the newly added classification head are trained. This is the most computationally efficient method and is a good starting point, especially when the fine-tuning dataset is small, to avoid overfitting. 

Full Fine-Tuning: This strategy involves unfreezing all the layers of the pre-trained model and training the entire network on the new dataset. 
 This allows the model to adjust its learned representations to the specifics of the new task. Full fine-tuning is more computationally expensive and generally requires a larger dataset to prevent the model from "forgetting" the valuable features learned during pre-training, a phenomenon known as catastrophic forgetting. 

Differential Learning Rates (Discriminative Fine-Tuning): This is a hybrid approach that recognizes that the layers of a pre-trained model learn features of varying complexity. The earlier layers tend to learn more general features, while later layers learn more specialized features. Therefore, it can be beneficial to use different learning rates for different parts of the model. 
 Typically, the newly added classification head is trained with a higher learning rate, while the pre-trained layers are fine-tuned with a much smaller learning rate. This allows for stable learning in the pre-trained parts of the model while enabling the new head to adapt more quickly to the new task. 

## Conceptual PyTorch Code Example
Here is a conceptual PyTorch code example that demonstrates loading a pre-trained model, adding a new classification head, and setting up an optimizer with differential learning rates.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel # A hypothetical pre-trained genomic model from Hugging Face

# 1. Load a pre-trained genomic foundational model
pre_trained_model = AutoModel.from_pretrained('hypothetical/genomic-bert')

# 2. Freeze the pre-trained layers (optional, for initial feature extraction phase)
for param in pre_trained_model.parameters():
    param.requires_grad = False

# 3. Define a new classification head
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# 4. Combine the pre-trained model with the new classification head
class FineTunedGenomicModel(nn.Module):
    def __init__(self, pre_trained_model, num_labels):
        super().__init__()
        self.pre_trained = pre_trained_model
        # Assuming the pre-trained model has a 'config.hidden_size' attribute
        self.classifier = ClassificationHead(pre_trained_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.pre_trained(input_ids=input_ids, attention_mask=attention_mask)
        # Use the representation of the [CLS] token for classification
        sequence_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(sequence_output)
        return logits

model = FineTunedGenomicModel(pre_trained_model, num_labels=2)

# Unfreeze the pre-trained layers for full fine-tuning with differential learning rates
for param in model.pre_trained.parameters():
    param.requires_grad = True

# 5. Configure an optimizer with differential learning rates
# Group parameters for the optimizer
optimizer_grouped_parameters = [
    {
        "params": model.pre_trained.parameters(),
        "lr": 1e-5,  # Smaller learning rate for the pre-trained part
    },
    {
        "params": model.classifier.parameters(),
        "lr": 1e-4,  # Larger learning rate for the new classification head
    },
]

optimizer = optim.AdamW(optimizer_grouped_parameters)

# The model can now be trained with this optimizer
```
## Interpreting 1D Convolutional Layers as Motif Detectors
In genomic models, particularly those with convolutional neural network (CNN) components, 1D convolutional layers (torch.nn.Conv1d) are adept at functioning as motif detectors. 
 DNA motifs are short, recurring patterns in DNA that are presumed to have a biological function, such as being binding sites for transcription factors.

### Function of a 1D Convolutional Layer
A 1D convolutional layer applies a set of filters (also called kernels) to an input sequence. 
 For genomic data, the input sequence is typically one-hot encoded, resulting in a matrix where the rows represent the four nucleotide bases (A, C, G, T) and the columns represent the sequence positions.

The torch.nn.Conv1d layer works as follows:

Filters (Kernels): Each filter is a small matrix of weights that is trained to recognize a specific pattern. The dimensions of a filter are (number of input channels, kernel size). In the case of one-hot encoded DNA, the number of input channels is 4. The kernel size determines the length of the motif the filter can detect. 

Dot Product Operation: The filter "slides" across the input sequence, and at each position, the dot product between the filter's weights and the corresponding subsequence of the input is computed. 

High Activation for Matching Patterns: If the subsequence at a particular position closely matches the pattern encoded in the filter's weights, the dot product will result in a high activation value. For instance, a filter weight that is high for 'A' at a certain position and low for other bases will be highly activated when it encounters an 'A' at that position in the input sequence.

This process is analogous to how Position Weight Matrices (PWMs) are used to score sequences for the presence of a motif. 
 The learned weights of the convolutional filters can be thought of as learning PWM-like representations of biologically relevant motifs. 

### Interpreting and Visualizing Learned Features
To understand what a trained model has learned, the filters of the convolutional layers can be extracted and visualized. 

Extracting Convolutional Filters: The weights of the trained torch.nn.Conv1d layer can be accessed from the model's state_dict(). These weights are a tensor of shape (number of filters, 4, kernel size).

Generating a Position Frequency Matrix (PFM): To create a PFM from a filter, one can find all the subsequences in a dataset that maximally activate that filter. 
 These activating subsequences are then aligned to create a frequency matrix, where each entry represents the count of each nucleotide at each position.

From PFM to PWM and Sequence Logos:

Position Weight Matrix (PWM): The PFM is then converted to a PWM by normalizing the frequencies at each position and applying a logarithmic transformation, often correcting for background nucleotide frequencies. 

Sequence Logos: A sequence logo is a graphical representation of a motif's PWM. 
 The height of each stack of letters represents the information content of that position in the motif, while the height of each letter within a stack is proportional to its frequency at that position. 
 This provides an intuitive visualization of the learned motif. 

By converting the learned filters into sequence logos, researchers can compare them to known motifs in databases like JASPAR, which can help in validating that the model is learning biologically meaningful features and can even lead to the discovery of novel motifs.
