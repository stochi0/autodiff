"""Automatic differentiation engine with neural network components."""

import numpy as np


class Value:
    """A scalar value with automatic differentiation support."""
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._op = _op
        self._children = set(_children)
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, _children={self._children}, _op={self._op}, label={self.label}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+', f"({self.label} + {other.label})")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), '-', f"({self.label} - {other.label})")
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*', f"({self.label} * {other.label})")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), '/', f"({self.label} / {other.label})")
        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / other.data**2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other.data, (self, other), '**', f"({self.label} ** {other.label})")
        def _backward():
            # d/dx (x^n) = n * x^(n-1)
            if abs(self.data) > 1e-10:
                self.grad += other.data * self.data**(other.data - 1) * out.grad
            # d/dn (x^n) = x^n * log(x) (only if x > 0)
            if self.data > 1e-10:
                other.grad += self.data**other.data * np.log(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), 'tanh', f"tanh({self.label})")
        def _backward():
            self.grad += (1 - np.tanh(self.data)**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, (self,), 'sigmoid', f"sigmoid({self.label})")
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu', f"relu({self.label})")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def leaky_relu(self, alpha=0.01):
        out = Value(max(alpha*self.data, self.data), (self,), 'leaky_relu', f"leaky_relu({self.label})")
        def _backward():
            self.grad += (out.data > 0) * out.grad + (out.data <= 0) * alpha * out.grad
        out._backward = _backward
        return out  

    def backward(self):
        """Backpropagate gradients through the computation graph."""
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)
        
        build_topo(self)
        
        # Reset all gradients in the computation graph
        for node in topo:
            node.grad = 0.0
        
        # Set gradient of root node to 1.0
        self.grad = 1.0
        
        # Backpropagate gradients in reverse topological order
        for node in reversed(topo):
            node._backward()

class Neuron:
    """A single neuron with weights, bias, and activation function."""
    
    def __init__(self, num_inputs, activation_function=lambda x: x):
        self.w = [Value(np.random.uniform(-1, 1), label=f"w{i}") for i in range(num_inputs)]
        self.b = Value(np.random.uniform(-1, 1), label="b")
        self.activation_function = activation_function

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # Compute dot product: w·x + b
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return self.activation_function(act)

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""
    
    def __init__(self, num_neurons, num_inputs, activation_function=lambda x: x):
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multi-layer perceptron (feedforward neural network)."""
    
    def __init__(self, num_inputs, num_hidden_layers, num_outputs):
        self.layers = []
        # First hidden layer: num_hidden_layers neurons, num_inputs inputs
        self.layers.append(Layer(num_hidden_layers, num_inputs, activation_function=lambda x: x.relu()))
        # Additional hidden layers: num_hidden_layers neurons, num_hidden_layers inputs
        for _ in range(num_hidden_layers - 1):
            self.layers.append(Layer(num_hidden_layers, num_hidden_layers, activation_function=lambda x: x.relu()))
        # Output layer: num_outputs neurons, num_hidden_layers inputs (linear activation for regression)
        self.layers.append(Layer(num_outputs, num_hidden_layers, activation_function=lambda x: x))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
