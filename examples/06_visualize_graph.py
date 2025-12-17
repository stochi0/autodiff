"""Visualize computation graphs using graphviz."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from beacongrad import Tensor, Linear, ReLU
from beacongrad.viz import draw_dot, save_graph

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

print("=" * 50)
print("Computation Graph Visualization")
print("=" * 50)

# Example 1: Simple arithmetic operations
print("\n1. Simple arithmetic operations")
a = Tensor([2.0, 3.0], requires_grad=True)
b = Tensor([1.0, 4.0], requires_grad=True)
c = a * b
d = c.sum()
d.backward()

print(f"a = {a.data}")
print(f"b = {b.data}")
print(f"c = a * b = {c.data}")
print(f"d = sum(c) = {d.item()}")
print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")

# Save graph
graph = draw_dot(d)
out = graph.render(os.path.join(ASSETS_DIR, "example_06_graph_simple"), format="svg", cleanup=True)
print(f"\nSaved graph to: {out}")

# Example 2: Neural network layer
print("\n2. Neural network layer")
x = Tensor([[1.0, 2.0]], requires_grad=True)
layer = Linear(2, 3)
y = layer(x)
z = y.sum()
z.backward()

print(f"Input x: {x.data}")
print(f"Output y: {y.data}")
print(f"Sum z: {z.item()}")

# Save graph
graph = draw_dot(z)
out = graph.render(os.path.join(ASSETS_DIR, "example_06_graph_neural_net"), format="svg", cleanup=True)
print(f"Saved graph to: {out}")

# Example 3: More complex computation
print("\n3. Complex computation with activations")
x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
y = x * 2
z = y.relu()
w = z.sum()
w.backward()

print(f"x = {x.data}")
print(f"y = x * 2 = {y.data}")
print(f"z = relu(y) = {z.data}")
print(f"w = sum(z) = {w.item()}")
print(f"x.grad = {x.grad}")

# Save graph
graph = draw_dot(w)
out = graph.render(os.path.join(ASSETS_DIR, "example_06_graph_activation"), format="svg", cleanup=True)
print(f"Saved graph to: {out}")

print("\n" + "=" * 50)
print("All graphs saved! Open the .svg files to view them.")
print("=" * 50)
