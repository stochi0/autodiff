from .nn import Value
import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation_function=lambda x: x):
        self.w = [Value(np.random.uniform(-1, 1), label=f"w{i}") for i in range(num_inputs)]
        self.b = Value(np.random.uniform(-1, 1), label="b")
        self.activation_function = activation_function

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.activation_function(sum((wi*xi for wi, xi in zip(self.w, x)), self.b))  # dot product of w and x + b

    def parameters(self):
        return self.w + [self.b]
