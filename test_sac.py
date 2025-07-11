import csv

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fractal_grid import FractalGrid
from sac_dqn_agent import SoftActorCriticAgent
from itertools import cycle
import networkx as nx
import json

SAVE_DIR = "save/"
NUM_EPISODES = 1000

plt.rcParams.update({
    'font.size': 20,  # Set default font size
    'font.weight': 'bold',  # Increase font weight slightly
    'axes.titlesize': 20,  # Title font size
    'axes.titleweight': 'bold',  # Title font weight
    'axes.labelsize': 20,  # Axis label font size
    'axes.labelweight': 'bold',  # Axis label font weight
    'xtick.labelsize': 20,  # X-axis tick label font size
    'ytick.labelsize': 20,  # Y-axis tick label font size
    'legend.fontsize': 20,  # Legend font size
    'lines.linewidth': 3,  # Default line thickness
    'axes.grid': False  # Disable grid lines
})


plt.rcParams['figure.dpi'] = 100

# Load Trained Model
def load_model(agent, actor_path, critic1_path, critic2_path):
    """Load the saved SAC model from the specified paths."""
    agent.load(actor_path, critic1_path, critic2_path)

def test_agent(env, agent, num_episodes=100):
    """Test the trained SAC agent without exploration (deterministic policy)."""
    time_steps = np.arange(0, 23 * 7, 1)  # Assuming a 7-day testing period with hourly time steps
    total_reward = 0

    # Data structures for collecting metrics
    num_microgrids = len(env.microgrids)
    net_load_data = [[] for _ in range(num_microgrids)]
    soc_data = [[] for _ in range(num_microgrids)]
    charge_discharge_data = [[] for _ in range(num_microgrids)]
    grid_data = [[] for _ in range(num_microgrids)]
    pv_dispatch_data = [[] for _ in range(num_microgrids)]
    switching_data = [[] for _ in range(num_microgrids)]
    power_transfer_data = [[] for _ in range(num_microgrids)]
    power_transfer_agg_data = [[] for _ in range(num_microgrids)]
    total_grid_power_data = []
    power_transfer_hourly = []
    total_pv_dispatch_data = []

    # Initialize switch dictionary
    switch_dict = {}

    for i, grid in enumerate(env.microgrids):
        for neighbor in grid.neighbors:
            switch_name = f"S_{min(grid.index, neighbor.index)}_to_{max(grid.index, neighbor.index)}"
            switch_dict[switch_name] = []

    for episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        power_transfer_daily = []
        hour = 0

        while not done:

            processed_switches = set()
            # Deterministic policy for testing
            action = agent.select_action(state, test_mode=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            episode_reward += reward

            # Collect power transfer data
            # each power transfer is (src, dst, power)
            # so for every hour, look for src and dst combo and sum power
            for power_transfer in _["power_transfers"]:
                src = power_transfer[0]
                dst = power_transfer[1]
                power = power_transfer[2]
                power_transfer_hourly.append([episode, hour, src, dst, power])
                # loop through power_transfer_daily and find src and dst combo and sum power
                for i in range(len(power_transfer_daily)):
                    if power_transfer_daily[i][0] == src and power_transfer_daily[i][1] == dst:
                        power_transfer_daily[i][2] += power
                        break
                else:
                    power_transfer_daily.append([src, dst, power])

            # Collect data for visualization
            total_grid_power_data.append(_["total_grid_exchange_power"])
            total_pv_dispatch_data.append(_["total_net_load"])

            for i, grid in enumerate(env.microgrids):
                net_load_data[i].append(_[f"net_load_{i}"])
                soc_data[i].append(_[f"soc_{i}"])
                charge_discharge_data[i].append(_[f"charge_discharge_{i}"])
                grid_data[i].append(_[f"grid_{i}"])
                pv_dispatch_data[i].append(_[f"pv_dispatch_{i}"])

                for neighbor in grid.neighbors:
                    switch_name = f"S_{min(grid.index, neighbor.index)}_to_{max(grid.index, neighbor.index)}"
                    if switch_name in switch_dict and switch_name not in processed_switches:
                        switch_dict[switch_name].append(_[switch_name])
                        processed_switches.add(switch_name)

                power_transfer_agg_data[i].append(_[f"power_transfer_agg_{i}"] if f"power_transfer_agg_{i}" in _ else 0)

            hour += 1

            state = next_state

        power_transfer_data.append(power_transfer_daily)

        print(f"Episode {episode + 1}/{num_episodes}: Total Reward: {episode_reward:.2f}")

    switching_data = [list(value) for value in switch_dict.values() if value is not None]
    # power transfers data is array of array with for each microgrid.
    # Within each microgrid, we have hourly data of power transfer to other microgrids that has day, hour, src, dst, power
    # save this to a CSV file with column names
    power_transfer_hourly_df = pd.DataFrame(power_transfer_hourly, columns=["episode", "hour", "src", "dst", "power"])
    power_transfer_hourly_df.to_csv("power_transfer_hourly.csv", index=False)

    # Convert data to arrays for visualization
    # plot_net_load(time_steps, net_load_data)
    # visualize_tree(env.microgrids)
    # plot_battery_soc(time_steps, charge_discharge_data, soc_data)
    # plot_grid_pv(time_steps, grid_data, pv_dispatch_data)
    # plot_switching_schedule(time_steps, switching_data)
    plot_power_transfer(time_steps, power_transfer_data)
    plot_power_transfer_agg(time_steps, power_transfer_agg_data)
    plot_power_transfer_together(time_steps, power_transfer_agg_data)
    # plot_total_grid_power_and_pv_dispatch(
    #    time_steps, total_grid_power_data[-161:], total_pv_dispatch_data[-161:]
    # )

# --- Plotting Functions ---


def plot_total_grid_power_and_pv_dispatch(time_steps, total_grid_power_data, total_pv_dispatch_data):
    plt.figure(figsize=(30, 8))

    # line styles for black-and-white compatibility
    line_styles = cycle([
        '-',
        '--',
        '-.',
        ':',
        (0, (3, 5, 1, 5)),  # Dash-dot-dash
        (0, (5, 10)),  # Long dashes
        (0, (1, 1)),  # Dense dots
        (0, (5, 2, 1, 2))  # Long dash followed by short dash
    ])

    # Plot the data
    plt.plot(time_steps, total_grid_power_data, label='Total power exchanged with Main Grid' , linestyle=next(line_styles))
    plt.plot(time_steps, total_pv_dispatch_data, label='Total Net MicroGrid power' , linestyle=next(line_styles))

    # Set x-axis ticks and labels to represent days
    xticks = time_steps[::24]  # Use every 24th hour for tick marks
    xtick_labels = [f'Day {t // 24}' for t in xticks]  # Convert tick labels to days
    plt.xticks(xticks, xtick_labels)

    # Set labels and title
    plt.xlabel('Time (days)')
    plt.ylabel('Power (kW)')
    plt.title('Total Grid Power and PV Dispatch')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(SAVE_DIR + "total_grid_power_and_pv_dispatch_sac.png")


import matplotlib.pyplot as plt


def plot_net_load(time_steps, net_load_data):
    plt.figure(figsize=(30, 8))

    # Plot the net load for each microgrid
    for i, net_load in enumerate(net_load_data):
        plt.plot(time_steps, net_load[-161:], label=f'Microgrid {i + 1}')

    # Set x-axis ticks and labels to represent days
    xticks = time_steps[::24]  # Use every 24th hour for tick marks
    xtick_labels = [f'Day {t // 24}' for t in xticks]  # Convert tick labels to days
    plt.xticks(xticks, xtick_labels)

    # Set labels and title
    plt.xlabel('Time (days)')
    plt.ylabel('Net Load (kWh)')
    plt.title('Net Load for Each Microgrid Over 7 Days')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(SAVE_DIR + "net_load_sac.png")


def plot_battery_soc(time_steps, charge_discharge_data, soc_data):
    n = len(charge_discharge_data)  # Number of microgrids
    cols = 2  # Number of columns
    rows = (n + 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(50, 8 * rows))  # Create grid layout

    for i, (charge_discharge, soc) in enumerate(zip(charge_discharge_data, soc_data)):
        row, col = divmod(i, cols)  # Get the correct position in the grid
        ax1 = axs[row, col] if rows > 1 else axs[col]  # Handle single-row layout

        # Bar chart for charging/discharging
        ax1.bar(time_steps, charge_discharge[-161:], label=f'Charging/Discharging {i + 1}', alpha=0.5)

        # Line plot for SOC
        ax2 = ax1.twinx()
        ax2.plot(time_steps, soc[-161:], label=f'SOC {i + 1}', marker='o', color='red')

        # Convert x-axis to represent days
        ax1.set_xticks(time_steps[::24])  # Set ticks every 24 hours
        ax1.set_xticklabels([f'Day {t // 24}' for t in time_steps[::24]])  # Convert ticks to days

        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Charging/Discharging Power (kW)')
        ax2.set_ylabel('SOC (%)')
        ax1.set_title(f'Microgrid {i + 1}')

    fig.tight_layout()

    plt.savefig(SAVE_DIR + "battery_soc_sac.png")


def plot_grid_pv(time_steps, grid_data, pv_dispatch_data):
    # Define line styles for black-and-white compatibility
    line_styles = cycle([
        '-',
        '--',
        '-.',
        ':',
        (0, (3, 5, 1, 5)),  # Dash-dot-dash
        (0, (5, 10)),  # Long dashes
        (0, (1, 1)),  # Dense dots
        (0, (5, 2, 1, 2))  # Long dash followed by short dash
    ])

    n = len(pv_dispatch_data)  # Number of microgrids
    cols = 2  # Number of columns
    rows = (n + 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(50, 8 * rows))  # Create grid layout

    for i, (pv_dispatch, grid_power) in enumerate(zip(pv_dispatch_data, grid_data)):
        row, col = divmod(i, cols)  # Get the correct position in the grid
        ax1 = axs[row, col] if rows > 1 else axs[col]  # Handle single-row layout

        # Bar chart for PV dispatch
        ax1.bar(time_steps, pv_dispatch[-161:], label=f'PV Dispatch {i + 1}', alpha=0.5)

        # Line plot for grid power bought/sold
        ax2 = ax1.twinx()
        ax2.plot(time_steps, grid_power[-161:], label=f'Grid Power {i + 1}', marker='x', color='green', linestyle=next(line_styles))

        # Convert x-axis to represent days
        ax1.set_xticks(time_steps[::24])  # Set ticks every 24 hours
        ax1.set_xticklabels([f'Day {t // 24}' for t in time_steps[::24]])  # Convert ticks to days

        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('PV Dispatch (kW)')
        ax2.set_ylabel('Grid Power (kW)')
        ax1.set_title(f'Microgrid {i + 1}')

    fig.tight_layout()

    plt.savefig(SAVE_DIR + "grid_pv_sac.png")


import matplotlib.pyplot as plt
import seaborn as sns

def plot_switching_schedule(time_steps, switching_data):
    plt.figure(figsize=(30, 8))  # Adjust figure size if necessary

    # Retain only the last 161 records in the value
    switching_data = [value[-161:] for value in switching_data]

    # Set x-axis ticks and labels to represent days
    xticks = time_steps[::24]  # Use every 24th hour for tick marks
    xtick_labels = [f'Day {t // 24}' for t in xticks]  # Convert tick labels to days

    # Create the heatmap
    sns.heatmap(switching_data, cmap='Blues', cbar=True, xticklabels=161,
                yticklabels=[f'Switch {i + 1}' for i in range(len(switching_data))])

    # Apply custom x-axis labels for days
    plt.xticks(ticks=xticks, labels=xtick_labels, rotation=45, ha='right')  # Rotate labels for clarity

    # Set labels and title
    plt.xlabel('Time (days)')
    plt.ylabel('Switch')
    plt.title('Switching Schedule Over 7 Days')

    # Save the plot
    plt.savefig(SAVE_DIR + "switching_schedule_sac.png")



def visualize_tree(nodes):
    """
    Visualize the fractal tree of nodes using networkx and matplotlib.
    :param nodes: List of all nodes.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Define graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    for node in nodes:
        G.add_node(node.index)
        for neighbor in node.neighbors:
            # Add labels to the edges
            edge_label = f"S_{min(node.index, neighbor.index)}_to_{max(node.index, neighbor.index)}"
            G.add_edge(node.index, neighbor.index, label=edge_label)

    # Plot settings
    plt.rcParams.update({
        'font.size': 20,  # Font size for labels
        'font.weight': 'bold',  # Bold font for labels
        'lines.linewidth': 3,  # Increase line thickness
    })

    # Draw the graph
    plt.figure(figsize=(10, 10))  # Larger figure size for clarity
    pos = nx.spring_layout(G, seed=42)  # Positioning for graph visualization
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightgreen',
        node_size=1200,  # Larger node size
        font_size=20,  # Larger font for node labels
        font_weight='bold',
        edge_color='black',
        width=3  # Thicker edges
    )

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=20,  # Larger font for edge labels
        font_weight='bold',
        font_color='red'
    )

    # Customize the title
    plt.title("FractalGrid Representation", fontsize=20, fontweight='bold')

    # Remove grid lines (though matplotlib usually does not add grid lines to these plots)
    plt.grid(False)

    # Save the figure
    plt.savefig(SAVE_DIR + "fractalgrid.png")

def plot_power_transfer(time_steps, power_transfer_data):

    # Define font and style settings
    plt.rcParams.update({
        'font.size': 20,  # Set font size for labels and titles
        'font.weight': 'bold',  # Use bold font for emphasis
        'axes.titlesize': 20,  # Title font size
        'axes.titleweight': 'bold',  # Title font weight
        'axes.labelsize': 20,  # Axis label font size
        'axes.labelweight': 'bold',  # Axis label font weight
        'xtick.labelsize': 20,  # X-axis tick label font size
        'ytick.labelsize': 20,  # Y-axis tick label font size
        'legend.fontsize': 20,  # Legend font size
        'lines.linewidth': 3,  # Default line thickness
        'axes.grid': False  # Disable grid lines
    })

    # Define layout: 2 columns and dynamic row adjustment
    n = 7  # Number of microgrids
    cols = 2  # Default column count
    rows = (n + 1) // cols  # Calculate rows needed

    # Special case: If n is odd, last subplot should be centered
    fig, axs = plt.subplots(rows, cols, figsize=(35, 10 * rows))

    # get last 7 days of data from power transfer
    power_transfer_data = power_transfer_data[-7:]

    for i, power_transfers in enumerate(power_transfer_data):
        row, col = divmod(i, cols)

        # Special case: If n is odd and last graph, center it
        if n % 2 == 1 and i == n - 1:
            ax = fig.add_subplot(rows, 1, rows)  # Full row width
        else:
            ax = axs[row, col] if rows > 1 else axs[col]

        # Create a directed graph for this microgrid
        G = nx.DiGraph()

        # Add directed edges with weights
        for src, dst, power in power_transfers:
            G.add_edge(src, dst, weight=round(power,2))

        # Generate layout with more spacing
        pos = nx.spring_layout(G,k=0.8, iterations=20)  # Increased k to spread nodes apart

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=1200, edgecolors="black", ax=ax)

        # Draw edges with directional arrows
        edges = G.edges(data=True)
        edge_colors = ['blue' for _ in edges]
        edge_widths = [max(0.5, d['weight'] / 5) for (_, _, d) in edges]  # Scale width

        edge_labels = {(u, v): f"{d['weight']:.2f} kW" for u, v, d in edges}

        nx.draw_networkx_edges(
            G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors,
            arrows=True, arrowsize=30, alpha=0.8,  # Slight transparency to avoid hiding details
            connectionstyle="arc3,rad=0.2", min_target_margin=15, ax=ax
        )

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color="red", ax=ax)

        # **Fix Node Label Overlaps**
        label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}  # Offset labels slightly
        nx.draw_networkx_labels(G, label_pos, font_size=14, font_weight="bold", ax=ax)

        ax.set_title(f"Day {i }")

        # Adjust layout and show the figure
        fig.tight_layout()
    plt.savefig("power_transfer_graphs.png")
    plt.show()


import matplotlib.pyplot as plt


def plot_power_transfer_together(time_steps, power_transfer_data):
    """
    Superimpose all 7 microgrid power transfer graphs into a single plot.

    :param time_steps: List of time steps.
    :param power_transfer_data: List of power transfer data for each microgrid.
    """
    # Set large font size and visualization settings
    plt.rcParams.update({
        'font.size': 40,  # Font size for axis labels and legends
        'font.weight': 'bold',
        'axes.titlesize': 40,
        'axes.titleweight': 'bold',
        'axes.labelsize': 40,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 40,
        'ytick.labelsize': 40,
        'legend.fontsize': 40,
        'lines.linewidth': 6,  # Increase line thickness for visibility
        'axes.grid': False
    })

    # Convert time_steps from hours to days
    time_in_days = [t / 23 for t in time_steps]

    # Create a single figure
    plt.figure(figsize=(50, 20))

    # Iterate over each microgrid and plot its power transfer on the same graph
    for i, power_transfer in enumerate(power_transfer_data):
        plt.plot(time_in_days[-161:], power_transfer[-161:], label=f'Microgrid {i} Transfer', linewidth=4)

    # Add labels, title, and legend
    plt.xlabel('Time (days)')
    plt.ylabel('Power Transfer (kW)')
    plt.title('Aggregated Power Transfer Across Microgrids')
    plt.legend()
    plt.grid(True)

    # Save and display the plot
    plt.savefig(SAVE_DIR + "power_transfer_sac_together.png")
    plt.show()

def plot_power_transfer_agg(time_steps, power_transfer_data):
    plt.rcParams.update({
        'font.size': 40,  # Set default font size
        'font.weight': 'bold',  # Increase font weight slightly
        'axes.titlesize': 40,  # Title font size
        'axes.titleweight': 'bold',  # Title font weight
        'axes.labelsize': 40,  # Axis label font size
        'axes.labelweight': 'bold',  # Axis label font weight
        'xtick.labelsize': 40,  # X-axis tick label font size
        'ytick.labelsize': 40,  # Y-axis tick label font size
        'legend.fontsize': 40,  # Legend font size
        'lines.linewidth': 6,  # Default line thickness
        'axes.grid': False  # Disable grid lines
    })

    n = len(power_transfer_data)  # Number of microgrids
    cols = 2  # Number of columns
    rows = (n + 1) // cols  # Calculate the number of rows needed

    # Convert time_steps from hours to days
    time_in_days = [t / 23 for t in time_steps]

    fig, axs = plt.subplots(rows, cols, figsize=(50, 8 * rows))  # Create grid layout

    for i, power_transfer in enumerate(power_transfer_data):
        row, col = divmod(i, cols)  # Get the correct position in the grid
        ax = axs[row, col] if rows > 1 else axs[col]  # Handle single-row layout

        # Line plot for power transfer
        ax.plot(time_in_days[-161:], power_transfer[-161:], label=f'Microgrid {i} Transfer')

        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Power Transfer (kW)')
        ax.set_title(f'Power Transfer for Microgrid {i}')
        ax.legend()
        ax.grid(True)

    fig.tight_layout()

    plt.savefig(SAVE_DIR + "power_transfer_sac.png")

# --- Main Code for Testing ---
if __name__ == "__main__":
    # Initialize the environment
    env = FractalGrid(num_microgrids=7)

    # Initialize the SAC agent
    agent = SoftActorCriticAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0]
    )

    # Load the trained SAC model
    actor_path = 'save/sac_fractal_grid_actor.pth'
    critic1_path = 'save/sac_fractal_grid_critic1.pth'
    critic2_path = 'save/sac_fractal_grid_critic2.pth'
    load_model(agent, actor_path, critic1_path, critic2_path)

    # Run the testing and generate plots
    test_agent(env, agent, NUM_EPISODES)
