from src.nn import Layer, Value
from src.viz import draw_dot

def main():
    # Create a layer with 3 neurons and 3 inputs (matching the neuron example)
    layer = Layer(3, 3, activation_function=lambda x: x.relu())
    
    # Create input values
    x1 = Value(1.0, label="x1")
    x2 = Value(2.0, label="x2")
    x3 = Value(3.0, label="x3")
    x = [x1, x2, x3]
    
    # Forward pass
    outputs = layer.forward(x)
    print(f"Layer outputs: {[out.data for out in outputs]}")
    
    # Combine all outputs into a single root value for visualization
    # This will create a graph that includes all neurons' computation graphs
    combined = outputs[0]
    combined.label = "neuron_0_output"
    for i, out in enumerate(outputs[1:], 1):
        combined = combined + out
        combined.label = f"layer_output"
    
    # Generate the graph
    dot = draw_dot(combined)
    dot.render('layer', format='svg', cleanup=True)
    print(f"Layer graph saved to layer.svg")
    print(f"Layer has {len(outputs)} neurons")
    print(f"Combined output value: {combined.data}")

if __name__ == "__main__":
    main()