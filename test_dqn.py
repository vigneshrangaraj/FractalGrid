import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fractal_grid import FractalGrid
from dqn_agent import ActorCriticAgent
import networkx as nx

SAVE_DIR = "save/"

# Load Trained Model
def load_model(agent, path):
    """Load the saved DQN model from the specified path."""
    agent.load(path)


def test_agent(env, agent, num_episodes=1):
    """Test the trained agent without exploration (greedy policy)."""
    time_steps = np.arange(0, 23, 1)  # Assuming 24-hour period
    total_reward = 0

    # Data to collect for each microgrid
    num_microgrids = len(env.microgrids)
    net_load_data = [[] for _ in range(num_microgrids)]
    soc_data = [[] for _ in range(num_microgrids)]
    charge_discharge_data = [[] for _ in range(num_microgrids)]
    grid_data = [[] for _ in range(num_microgrids)]
    pv_dispatch_data = [[] for _ in range(num_microgrids)]
    switching_data = [[] for _ in range(num_microgrids)]
    power_transfer_data = [[] for _ in range(num_microgrids)]
    total_grid_power_data = []
    total_pv_dispatch_data = []

    # Initialize the switch dictionary
    switch_dict = {}

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        # Iterate through each microgrid and neighbor to populate switch_dict
        for i, grid in enumerate(env.microgrids):
            for j, neighbor in enumerate(grid.neighbors):
                switch_name = f"S_{min(grid.index, neighbor.index)}_to_{max(grid.index, neighbor.index)}"
                switch_dict[switch_name] = []

        while not done:
            # Act greedily (no exploration) using trained policy
            action = agent.act(state, test_mode=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            processed_switches = set()

            # collect grid power and total pv dispatch
            total_grid_power_data.append(_["total_grid_exchange_power"])
            total_pv_dispatch_data.append(_["total_pv_dispatch_power"])

            for i, grid in enumerate(env.microgrids):
                net_load_data[i].append(_[f"net_load_{i}"])
                soc_data[i].append(_[f"soc_{i}"])
                charge_discharge_data[i].append(_[f"charge_discharge_{i}"])
                grid_data[i].append(_[f"grid_{i}"])
                pv_dispatch_data[i].append(_[f"pv_dispatch_{i}"])

                # Loop through neighbors and add unique switch states from the dictionary
                for j, neighbor in enumerate(grid.neighbors):
                    switch_name = f"S_{min(grid.index, neighbor.index)}_to_{max(grid.index, neighbor.index)}"
                    if switch_name in switch_dict and switch_name not in processed_switches:
                        switch_dict[switch_name].append(_[f"S_{min(grid.index, neighbor.index)}_to_{max(grid.index, neighbor.index)}"])
                        processed_switches.add(switch_name)

                power_transfer_data[i].append(_[f"power_transfer_{i}"] if f"power_transfer_{i}" in _ else 0)

            state = next_state

        print(f"Total Reward: {total_reward}")

    switching_data = [list(value) for value in switch_dict.values() if value is not None]

    # Convert lists to arrays for plotting
    visualize_tree(env.microgrids)
    plot_net_load(time_steps, net_load_data)
    plot_battery_soc(time_steps, charge_discharge_data, soc_data)
    plot_grid_pv(time_steps, grid_data, pv_dispatch_data)
    plot_switching_schedule(time_steps, switching_data)
    plot_power_transfer(time_steps, power_transfer_data)
    plot_total_grid_power_and_pv_dispatch(time_steps, total_grid_power_data, total_pv_dispatch_data)



# --- Plotting Functions ---
def plot_total_grid_power_and_pv_dispatch(time_steps, total_grid_power_data, total_pv_dispatch_data):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, total_grid_power_data, label='Total Grid Power')
    plt.plot(time_steps, total_pv_dispatch_data, label='Total PV Dispatch')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('Total Grid Power and PV Dispatch')
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_DIR + "total_grid_power_and_pv_dispatch.png")

def plot_net_load(time_steps, net_load_data):
    plt.figure(figsize=(10, 6))
    for i, net_load in enumerate(net_load_data):
        plt.plot(time_steps, net_load, label=f'Microgrid {i + 1}')
    plt.xlabel('Time (hours)')
    plt.ylabel('Net Load (kWh)')
    plt.title('Net Load for Each Microgrid Over 24 Hours')
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_DIR + "net_load.png")


def plot_battery_soc(time_steps, charge_discharge_data, soc_data):
    n = len(charge_discharge_data)  # Number of microgrids
    cols = 2  # Number of columns
    rows = (n + 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(15, 6 * rows))  # Create grid layout

    for i, (charge_discharge, soc) in enumerate(zip(charge_discharge_data, soc_data)):
        row, col = divmod(i, cols)  # Get the correct position in the grid
        ax1 = axs[row, col]

        # Bar chart for charging/discharging
        ax1.bar(time_steps, charge_discharge, label=f'Charging/Discharging {i + 1}', alpha=0.5)

        # Line plot for SOC
        ax2 = ax1.twinx()
        ax2.plot(time_steps, soc, label=f'SOC {i + 1}', marker='o', color='red')

        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Charging/Discharging Power (kW)')
        ax2.set_ylabel('SOC (%)')
        ax1.set_title(f'Microgrid {i + 1}')

    fig.tight_layout()

    plt.savefig(SAVE_DIR + "battery_soc.png")

def plot_grid_pv(time_steps, grid_data, pv_dispatch_data):
    n = len(pv_dispatch_data)  # Number of microgrids
    cols = 2  # Number of columns
    rows = (n + 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(15, 6 * rows))  # Create grid layout

    for i, (pv_dispatch, grid_power) in enumerate(zip(pv_dispatch_data, grid_data)):
        row, col = divmod(i, cols)  # Get the correct position in the grid
        ax1 = axs[row, col]

        # Bar chart for PV dispatch
        ax1.bar(time_steps, pv_dispatch, label=f'PV Dispatch {i + 1}', alpha=0.5)

        # Line plot for grid power bought/sold
        ax2 = ax1.twinx()
        ax2.plot(time_steps, grid_power, label=f'Grid Power {i + 1}', marker='x', color='green')

        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('PV Dispatch (kW)')
        ax2.set_ylabel('Grid Power (kW)')
        ax1.set_title(f'Microgrid {i + 1}')

    fig.tight_layout()

    plt.savefig(SAVE_DIR + "grid_pv.png")



def plot_switching_schedule(time_steps, switching_data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(switching_data, cmap='Blues', cbar=True, xticklabels=time_steps,
                yticklabels=[f'Switch {i + 1}' for i in range(len(switching_data))])
    plt.xlabel('Time (hours)')
    plt.ylabel('Switch')
    plt.title('Switching Schedule Over 24 Hours')
    plt.savefig(SAVE_DIR + "switching_schedule.png")

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
            G.add_edge(node.index, neighbor.index)

    # Draw the graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)  # Positioning for graph visualization
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=1000, font_size=8, font_weight='bold', edge_color='black')
    plt.title("FractalGrid Representation")
    plt.savefig(SAVE_DIR + "fractalgrid.png")

def plot_power_transfer(time_steps, power_transfer_data):
    n = len(power_transfer_data)  # Number of microgrids
    cols = 2  # Number of columns
    rows = (n + 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(15, 6 * rows))  # Create grid layout

    for i, power_transfer in enumerate(power_transfer_data):
        row, col = divmod(i, cols)  # Get the correct position in the grid
        ax = axs[row, col]

        # Line plot for power transfer
        ax.plot(time_steps, power_transfer, label=f'Microgrid {i + 1} Transfer')

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power Transfer (kW)')
        ax.set_title(f'Power Transfer for Microgrid {i + 1}')
        ax.legend()
        ax.grid(True)

    fig.tight_layout()

    plt.savefig(SAVE_DIR + "power_transfer.png")


# --- Main Code for Testing ---
if __name__ == "__main__":
    # Initialize the environment
    env = FractalGrid(num_microgrids=7)  # Adjust initialization as needed
    
    # Initialize the DQN agent
    agent = ActorCriticAgent(env.observation_space, env.action_space, gamma=0.99)
    
    # Load the trained model
    model_path = 'save/dqn_fractal_grid.pth'
    load_model(agent, model_path)
    
    # Run the testing and generate plots
    test_agent(env, agent, 1)
