"""Visualize decision boundary of a trained classifier."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from beacongrad import Tensor, MLP, CrossEntropyLoss, Adam

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

print("=" * 50)
print("Decision Boundary Visualization")
print("=" * 50)

# Generate 2D classification data (moon-like dataset)
np.random.seed(42)
n_samples = 300

# Create two classes with different distributions
X1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([-1, 0])
X2 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([1, 0])
X = np.vstack([X1, X2]).astype(np.float32)
y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(np.int64)

# Shuffle
indices = np.random.permutation(n_samples)
X = X[indices]
y = y[indices]

print(f"Data: X shape={X.shape}, y shape={y.shape}")

# Build model
model = MLP(
    input_size=2,
    hidden_sizes=[16, 16],
    output_size=2,
    activation="relu",
    dropout=0.0,
)

print(f"\nModel: {model}")
print(f"Number of parameters: {sum(p.size for p in model.parameters())}")

# Loss and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
epochs = 200
losses = []

for epoch in range(epochs):
    X_tensor = Tensor(X)
    logits = model(X_tensor)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 50 == 0:
        predictions = np.argmax(logits.data, axis=1)
        accuracy = (predictions == y).mean()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Final evaluation
model.eval()
X_tensor = Tensor(X)
logits = model(X_tensor)
predictions = np.argmax(logits.data, axis=1)
accuracy = (predictions == y).mean()
print(f"\nFinal Accuracy: {accuracy:.4f}")

# Create decision boundary plot
print("\nGenerating decision boundary plot...")

# Create a mesh grid
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh grid
mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
mesh_tensor = Tensor(mesh_points)
model.eval()
mesh_logits = model(mesh_tensor)
mesh_pred = np.argmax(mesh_logits.data, axis=1)
Z = mesh_pred.reshape(xx.shape)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Decision boundary
ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', edgecolors='black', label='Class 0')
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s', edgecolors='black', label='Class 1')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Decision Boundary')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Training loss
ax2.plot(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(ASSETS_DIR, "example_07_decision_boundary_binary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved decision boundary plot to: {out_path}")

# Also create a 3-class example
print("\nGenerating 3-class decision boundary plot...")

# Generate 3-class data
np.random.seed(42)
n_per_class = 100
X1 = np.random.randn(n_per_class, 2) * 0.4 + np.array([0, 1])
X2 = np.random.randn(n_per_class, 2) * 0.4 + np.array([-1, -1])
X3 = np.random.randn(n_per_class, 2) * 0.4 + np.array([1, -1])
X_multi = np.vstack([X1, X2, X3]).astype(np.float32)
y_multi = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), np.full(n_per_class, 2)]).astype(np.int64)

indices = np.random.permutation(len(X_multi))
X_multi = X_multi[indices]
y_multi = y_multi[indices]

# Build and train model
model_multi = MLP(
    input_size=2,
    hidden_sizes=[16, 16],
    output_size=3,
    activation="relu",
    dropout=0.0,
)

optimizer_multi = Adam(model_multi.parameters(), lr=0.01)
model_multi.train()

for epoch in range(200):
    X_tensor = Tensor(X_multi)
    logits = model_multi(X_tensor)
    loss = criterion(logits, y_multi)
    
    optimizer_multi.zero_grad()
    loss.backward()
    optimizer_multi.step()

# Create mesh for 3-class
x_min, x_max = X_multi[:, 0].min() - 1, X_multi[:, 0].max() + 1
y_min, y_max = X_multi[:, 1].min() - 1, X_multi[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
mesh_tensor = Tensor(mesh_points)
model_multi.eval()
mesh_logits = model_multi(mesh_tensor)
mesh_pred = np.argmax(mesh_logits.data, axis=1)
Z = mesh_pred.reshape(xx.shape)

# Plot 3-class
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
ax.scatter(X_multi[y_multi == 0, 0], X_multi[y_multi == 0, 1], c='blue', marker='o', edgecolors='black', label='Class 0')
ax.scatter(X_multi[y_multi == 1, 0], X_multi[y_multi == 1, 1], c='red', marker='s', edgecolors='black', label='Class 1')
ax.scatter(X_multi[y_multi == 2, 0], X_multi[y_multi == 2, 1], c='green', marker='^', edgecolors='black', label='Class 2')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('3-Class Decision Boundary')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(ASSETS_DIR, "example_07_decision_boundary_3class.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved 3-class decision boundary plot to: {out_path}")

print("\n" + "=" * 50)
print("All visualizations saved!")
print("=" * 50)
