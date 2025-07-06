import networkx as nx
import matplotlib.pyplot as plt

# Example microgrid connections with power transfer values
power_transfers = [
    (1, 3, 5),   # Microgrid 1 sends 5 kW to Microgrid 3
    (2, 5, 12),  # Microgrid 2 sends 12 kW to Microgrid 5
    (2, 6, 7),   # Microgrid 2 sends 7 kW to Microgrid 6
    (2, 4, 5)    # Microgrid 2 sends 5 kW to Microgrid 4
]

# Create a directed graph
G = nx.DiGraph()

# Add directed edges with weights
for src, dst, power in power_transfers:
    G.add_edge(src, dst, weight=power)

# Generate layout for positioning
pos = nx.spring_layout(G, seed=42)

# Draw nodes
plt.figure(figsize=(16, 8))
nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=1000, edgecolors="black")

# Draw edges with directional arrows
edges = G.edges(data=True)
edge_colors = ['blue' for _ in edges]
edge_widths = [max(0.5, d['weight'] / 5) for (_, _, d) in edges]  # Ensure minimum width for visibility

nx.draw_networkx_edges(
    G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors,
    arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1", min_target_margin=10
)

# Add labels for microgrids
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Add edge labels with power transfer values
edge_labels = {(u, v): f"{d['weight']} kW" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red")

plt.title("Directed Power Transfer in Fractal Microgrid System")
plt.show()