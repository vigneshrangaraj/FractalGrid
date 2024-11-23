import matplotlib.pyplot as plt
import networkx as nx

def draw_actor_critic_network():
    plt.figure(figsize=(14, 10))

    # Define layers for Actor and Critic
    layers = {
        "Actor Input": ["S1", "S2", "S3"],
        "Actor Hidden 1": ["256 Neurons (H1)"],
        "Actor Hidden 2": ["256 Neurons (H2)"],
        "Actor Output": ["A1", "A2", "A3"],
        "Critic Input": ["S1", "S2", "S3", "A1", "A2", "A3"],
        "Critic Hidden 1": ["256 Neurons (CH1)"],
        "Critic Hidden 2": ["256 Neurons (CH2)"],
        "Critic Output": ["Q"]
    }

    # Create a graph
    G = nx.DiGraph()

    # Add nodes for all layers
    for layer_name, nodes in layers.items():
        for node in nodes:
            G.add_node(node, layer=layer_name)

    # Actor connections
    for src in layers["Actor Input"]:
        for dst in layers["Actor Hidden 1"]:
            G.add_edge(src, dst)
    for src in layers["Actor Hidden 1"]:
        for dst in layers["Actor Hidden 2"]:
            G.add_edge(src, dst)
    for src in layers["Actor Hidden 2"]:
        for dst in layers["Actor Output"]:
            G.add_edge(src, dst)

    # Critic connections
    for src in layers["Critic Input"]:
        for dst in layers["Critic Hidden 1"]:
            G.add_edge(src, dst)
    for src in layers["Critic Hidden 1"]:
        for dst in layers["Critic Hidden 2"]:
            G.add_edge(src, dst)
    for src in layers["Critic Hidden 2"]:
        for dst in layers["Critic Output"]:
            G.add_edge(src, dst)

    # Interconnect Actor Output with Critic Input
    for actor_output in layers["Actor Output"]:
        for critic_input in layers["Critic Input"][-3:]:  # Last 3 nodes are actions
            if actor_output == critic_input:
                G.add_edge(actor_output, critic_input)

    # Define node positions (left-to-right layout for clarity)
    pos = {}
    layer_x_positions = {
        "Actor Input": 1,
        "Actor Hidden 1": 2,
        "Actor Hidden 2": 3,
        "Actor Output": 4,
        "Critic Input": 6,
        "Critic Hidden 1": 7,
        "Critic Hidden 2": 8,
        "Critic Output": 9
    }
    for layer_name, x_position in layer_x_positions.items():
        nodes = layers[layer_name]
        y_positions = range(len(nodes))
        for i, node in enumerate(nodes):
            pos[node] = (x_position, -i * 1.5)

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=15,
        edge_color="black"
    )

    # Add labels and title
    plt.title("Actor-Critic Neural Network Architecture (256 Hidden Neurons)", fontsize=16)
    plt.tight_layout()
    plt.savefig("actor_critic_network_updated.png", dpi=300)
    plt.show()

# Call the function to draw the updated network
draw_actor_critic_network()
