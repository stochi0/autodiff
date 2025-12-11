from nn import MLP, Value


def train(mlp, x, y, epochs=100, learning_rate=0.01):
    """Train an MLP using gradient descent.
    
    Args:
        mlp: Multi-layer perceptron to train
        x: List of input samples (each sample is a list of Value objects)
        y: List of target outputs (each target is a list of Value objects)
        epochs: Number of training epochs
        learning_rate: Learning rate for gradient descent
    
    Returns:
        Final loss value
    """
    for epoch in range(epochs):
        # Forward pass: compute loss
        loss = Value(0.0)
        for x_i, y_i in zip(x, y):
            output = mlp(x_i)
            # Compute MSE loss for each output element
            for out_val, target_val in zip(output, y_i):
                loss += (out_val - target_val) ** Value(2)
        loss = loss / Value(len(x))
        
        print(f"Epoch {epoch}, Loss: {loss.data:.6f}")
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update parameters using gradient descent
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
    
    return loss.data


if __name__ == "__main__":
    # Create MLP: 3 inputs, 2 hidden neurons, 1 output
    mlp = MLP(3, 2, 1)
    
    # Training data
    x = [
        [Value(1), Value(2), Value(3)],
        [Value(4), Value(5), Value(6)],
        [Value(7), Value(8), Value(9)]
    ]
    y = [
        [Value(1)],
        [Value(2)],
        [Value(3)]
    ]
    
    # Train the model
    loss = train(mlp, x, y, epochs=100, learning_rate=0.01)
    print(f"Final Loss: {loss:.6f}")