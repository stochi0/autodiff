"""
Optimization algorithms (similar to torch.optim).

Provides various optimizers for training neural networks.
"""

import numpy as np
from typing import List
from .tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, parameters: List[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """Zero all gradients."""
        for p in self.parameters:
            if p.requires_grad:
                p.grad = np.zeros_like(p.data)

    def step(self):
        """Perform a single optimization step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement step()")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters: List of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: L2 regularization factor (default: 0)
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize velocity for momentum
        self.velocities = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        """Perform parameter update."""
        for i, p in enumerate(self.parameters):
            if not p.requires_grad or p.grad is None:
                continue

            grad = p.grad

            # Add weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Update velocity and parameter
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * grad
