# brew install graphviz
# pip install graphviz
# sudo apt-get install graphviz

from graphviz import Digraph
from engine import Value


def _trace(root):
    # Initialize sets to store unique nodes and edges of the computational graph
    nodes, edges = set(), set()
    
    def build(v):
        # Inner recursive function to build the graph structure
        # If we haven't seen this node before
        if v not in nodes:
            # Add the current node to our set of nodes
            nodes.add(v)
            # Iterate through all predecessor nodes (parents in computational graph)
            for child in v._prev:
                # Add an edge from child to current node
                edges.add((child, v))
                # Recursively process the child node
                build(child)
    
    # Start the graph building process from the root node
    build(root)
    return nodes, edges


def draw_dot(root, format='svg', rankdir='BT'):
    """
    Creates a graphviz visualization of the computational graph
    In this graph, circles are the nodes that contain values, while squares contain math operations.
    
    Parameters:
    root: The root node of the computational graph
    format: Output format for the graph (e.g., 'svg', 'png')
    rankdir: Direction of graph layout
        'LR' = Left to Right
        'TB' = Top to Bottom
    """
    assert type(root) is Value, "The root object must be an instance of the Value class"
    # Validate the rankdir parameter
    assert rankdir in ['LR', 'TB', 'BT']
    
    # Get the graph structure using trace function
    nodes, edges = _trace(root)
    
    # Initialize a new Digraph (directed graph) object with specified format and layout direction
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    # Process each node in the graph
    for n in nodes:
        # Create a node in the graph with:
        # - Unique identifier based on object's memory address (id(n))
        # - Label showing node's data and gradient values in a record format
        # - Shape set to 'record' for structured display
        dot.node(name=str(id(n)), 
                label = f"{{ {n.label} | data {n.data:.2f} | grad {n.grad:.3f} }}" if n.label is not None \
                   else f"{{ data {n.data:.2f} | grad {n.grad:.3f} }}",
                shape='record')
        
        # If the node has an operation (_op), create an additional node for the operation
        if n._op:
            # Create operation node
            dot.node(name=str(id(n)) + n._op, label=n._op)
            # Connect operation node to result node
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    # Process each edge in the graph
    for n1, n2 in edges:
        # Connect nodes with their operations
        # n1 (input) -> n2._op (operation) -> n2 (output)
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
