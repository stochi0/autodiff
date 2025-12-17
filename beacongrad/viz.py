"""Visualization utilities for computation graphs and training."""

import numpy as np
from typing import Set, Optional
from graphviz import Digraph
from .tensor import Tensor


def trace(root: Tensor) -> tuple[Set[Tensor], Set[tuple[Tensor, Tensor]]]:
    """
    Build a set of all nodes and edges in a computation graph.
    
    Returns:
        nodes: Set of all tensors in the graph
        edges: Set of (parent, child) tuples representing edges
    """
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_dot(root: Tensor, format: str = "svg", rankdir: str = "LR") -> Digraph:
    """
    Draw a computation graph using graphviz.
    
    Args:
        root: Root tensor to visualize
        format: Output format (svg, png, pdf, etc.)
        rankdir: Graph direction (LR for left-to-right, TB for top-to-bottom)
    
    Returns:
        graphviz Digraph object
    """
    nodes, edges = trace(root)
    
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        # Create a unique ID for each node
        uid = str(id(n))
        
        # Format data value
        data_str = format_tensor_data(n.data)
        
        # Format gradient if available
        grad_str = ""
        if n.grad is not None:
            grad_str = f"\\ngrad={format_tensor_data(n.grad)}"
        
        # Format shape
        shape_str = f"\\nshape={n.shape}" if n.ndim > 0 else ""
        
        # Create label
        label = f"{data_str}{grad_str}{shape_str}"
        if n._op:
            label = f"{n._op}\\n{label}"
        
        # Color based on whether it requires grad
        color = "lightblue" if n.requires_grad else "lightgray"
        
        dot.node(name=uid, label=label, fillcolor=color, style="filled")
    
    # Add edges
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))
    
    return dot


def format_tensor_data(data: np.ndarray, max_len: int = 10) -> str:
    """Format tensor data for display in graph nodes."""
    flat = data.flatten()
    
    if flat.size == 0:
        return "[]"
    elif flat.size == 1:
        return f"{flat[0]:.4f}"
    elif flat.size <= max_len:
        # Show all values
        vals = ", ".join(f"{x:.3f}" for x in flat)
        return f"[{vals}]"
    else:
        # Show first few and last few
        first = ", ".join(f"{x:.3f}" for x in flat[:3])
        last = ", ".join(f"{x:.3f}" for x in flat[-3:])
        return f"[{first}, ..., {last}]"


def save_graph(root: Tensor, filename: str = "graph", format: str = "svg") -> str:
    """
    Save computation graph to file.
    
    Args:
        root: Root tensor to visualize
        filename: Output filename (without extension)
        format: Output format (svg, png, pdf, etc.)
    
    Returns:
        Path to saved file
    """
    dot = draw_dot(root, format=format)
    output_path = dot.render(filename, cleanup=True)
    return output_path
