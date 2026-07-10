import matplotlib.pyplot as plt
import seaborn as sns

# Use the attn_weights from the previous example
# Shape: (batch_size, seq_length, seq_length)
# Let's visualize the weights for the first sequence in the batch.
attention_matrix = attn_weights[...](asc_slot://start-slot-94).detach().cpu().numpy()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attention_matrix, cmap='viridis')
plt.title('Attention Heatmap')
plt.xlabel('Key Positions (Attended To)')
plt.ylabel('Query Positions (Attending From)')
plt.show()
