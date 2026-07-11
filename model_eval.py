import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Example ground truth labels and model prediction scores
# An imbalanced dataset with 5 positive samples out of 20 (25%)
y_true = np.array()
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.3, 0.4, 0.1, 0.05, 0.15, 0.7, 0.5, 0.6, 0.2, 0.3, 0.4, 0.9, 0.3, 0.75, 0.25])

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# --- Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)

# --- Plotting ---
plt.figure(figsize=(12, 5))

# Plot ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot(, , color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
# Plot the no-skill line (prevalence of the positive class)
no_skill = len(y_true[y_true==1]) / len(y_true)
plt.plot([...](asc_slot://start-slot-262), [no_skill, no_skill], linestyle='--', color='navy', label='No-Skill')
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

print(f"AUC-ROC: {roc_auc:.4f}")
print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
