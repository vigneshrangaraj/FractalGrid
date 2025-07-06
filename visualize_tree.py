import networkx as nx
import matplotlib.pyplot as plt

from fractal_grid import FractalGrid

SAVE_DIR = "save/"


def visualize_tree(nodes):
    """
    Visualize the fractal tree of nodes using networkx and matplotlib.
    :param nodes: List of all nodes.
    """
    G = nx.Graph()

    # Add nodes and edges to the graph
    for node in nodes:
        G.add_node(node.index)
        for neighbor in node.neighbors:
            # Add labels to the edges
            edge_label = f"S_{min(node.index, neighbor.index)}_to_{max(node.index, neighbor.index)}"
            G.add_edge(node.index, neighbor.index, label=edge_label)

    # Draw the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # Positioning for graph visualization
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=1000, font_size=24, font_weight='bold', edge_color='black')

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=24, font_color='red')

    plt.title("FractalGrid Representation")
    plt.savefig(SAVE_DIR + "fractalgrid3.png")


env = FractalGrid(num_microgrids=7)
visualize_tree(env.microgrids)