import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Dataset: ML model training simulation
epochs = list(range(1, 21))

train_loss = [
    2.5 * np.exp(-0.15 * e) + 0.05 * np.random.randn()
    for e in epochs
]

val_loss = [
    2.8 * np.exp(-0.12 * e) + 0.1 * np.random.randn()
    for e in epochs
]

train_acc = [
    1 - 0.9 * np.exp(-0.2 * e) + 0.01 * np.random.randn()
    for e in epochs
]

val_acc = [
    1 - 0.95 * np.exp(-0.18 * e) + 0.015 * np.random.randn()
    for e in epochs
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

fig.suptitle(
    'Model Training Dashboard',
    fontsize=13,
    fontweight='bold'
)

# Loss curves
ax1.plot(
    epochs,
    train_loss,
    'b-o',
    label='Train Loss',
    markersize=4
)

ax1.plot(
    epochs,
    val_loss,
    'r-s',
    label='Val Loss',
    markersize=4
)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy curves
ax2.plot(
    epochs,
    train_acc,
    'b-o',
    label='Train Acc',
    markersize=4
)

ax2.plot(
    epochs,
    val_acc,
    'r-s',
    label='Val Acc',
    markersize=4
)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training & Validation Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()

# Save the visualization
plt.savefig('training_curves.png', dpi=80)

print('Plot saved successfully.')
print(f'Final Train Acc: {train_acc[-1]:.3f}')
print(f'Final Val Acc:   {val_acc[-1]:.3f}')