import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel # A popular library for pre-trained models
# You may need to install: pip install transformers peft
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. Load a pre-trained model and add a classification head ---
# In a real scenario, this would be a model like 'armheb/dna_bert_6'
# For demonstration, we'll use a generic BERT model.
try:
    base_model = AutoModel.from_pretrained("bert-base-uncased")
except OSError: # Handle case where user is offline
    from transformers import BertConfig, BertModel
    base_model = BertModel(BertConfig())


class FineTunedGenomicModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base = base_model
        # New head for a binary classification task
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits

model = FineTunedGenomicModel(base_model, num_labels=2)

# --- 2. Configure Optimizer with Differential Learning Rates ---
optimizer_grouped_parameters = [
    {"params": model.base.parameters(), "lr": 1e-5}, # Smaller LR for base
    {"params": model.classifier.parameters(), "lr": 1e-4}, # Larger LR for head
]
optimizer = optim.AdamW(optimizer_grouped_parameters)
print("Optimizer configured with differential learning rates.")

# --- 3. Configure LoRA for Parameter-Efficient Fine-Tuning ---
# Define the LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=8,  # The rank of the update matrices (a small number)
    lora_alpha=32, # A scaling factor for the LoRA weights
    lora_dropout=0.1,
    target_modules=["query", "key"] # Target the query and key matrices in attention layers
)

# Create a PEFT model with LoRA
peft_model = get_peft_model(model, lora_config)

print("\nTrainable parameters after applying LoRA:")
peft_model.print_trainable_parameters()

# You can now train 'peft_model' as you would a regular PyTorch model.
# Only the LoRA adapter parameters will be updated.
