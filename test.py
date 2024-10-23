import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id):
        self.id = id
        self.connections = []

    def add_connection(self, other_node):
        """Create a bidirectional edge between nodes"""
        self.connections.append(other_node)
        other_node.connections.append(self)

    def __repr__(self):
        return f"Node({self.id})"

def fractal_tree(node_count, parent=None, created_nodes=0):
    """
    Recursively create nodes and edges based on a fractal tree structure.
    :param node_count: Total number of nodes to create.
    :param parent: The parent node to connect with.
    :param created_nodes: The number of nodes already created.
    :return: A list of nodes and the number of nodes created so far.
    """
    if created_nodes >= node_count:
        return [], created_nodes

    if parent is None:
        # Create root node if parent doesn't exist
        parent = Node(id="Root")
        created_nodes += 1
        nodes = [parent]
    else:
        nodes = []

    # Recursively create branches (children) until node_count is reached
    left_child = Node(id=f"{parent.id}_left")
    right_child = Node(id=f"{parent.id}_right")

    # Connect parent to children
    parent.add_connection(left_child)
    parent.add_connection(right_child)

    nodes.extend([left_child, right_child])
    created_nodes += 2

    if created_nodes >= node_count:
        return nodes, created_nodes

    # Recursively build the tree on the left and right children
    new_nodes, created_nodes = fractal_tree(node_count, left_child, created_nodes)
    nodes.extend(new_nodes)

    if created_nodes < node_count:
        new_nodes, created_nodes = fractal_tree(node_count, right_child, created_nodes)
        nodes.extend(new_nodes)

    return nodes, created_nodes

def create_tree(node_count):
    """
    Create the grid based on a fractal tree structure with a given number of nodes.
    :param node_count: The number of nodes in the fractal tree.
    :return: List of all nodes created.
    """
    nodes, _ = fractal_tree(node_count)
    return nodes

def visualize_tree(nodes):
    """
    Visualize the fractal tree of nodes using networkx and matplotlib.
    :param nodes: List of all nodes.
    """
    G = nx.Graph()

    # Add nodes and edges to the graph
    for node in nodes:
        G.add_node(node.id)
        for neighbor in node.connections:
            G.add_edge(node.id, neighbor.id)

    # Draw the graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)  # Positioning for graph visualization
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=1000, font_size=8, font_weight='bold', edge_color='black')
    plt.title("Fractal Tree Structure Graph Representation")
    plt.show()

# Example usage
if __name__ == "__main__":
    node_count = 7  # Set the total number of nodes you want to create
    nodes = create_tree(node_count)

    # Visualize the fractal tree graph
    visualize_tree(nodes)
