import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

SAVE_DIR = "save/"

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

# line styles for black-and-white compatibility
line_styles = cycle([
    '-',
    '--',
    '-.',
    ':',
    (0, (3, 5, 1, 5)),  # Dash-dot-dash
    (0, (5, 10)),       # Long dashes
    (0, (1, 1)),        # Dense dots
    (0, (5, 2, 1, 2))   # Long dash followed by short dash
])

# Load the DDPG metrics
with open("save/metrics.json", "r") as file:
    ddpg_metrics = json.load(file)

# Load the SAC metrics
with open("save/metrics_sac.json", "r") as file:
    sac_metrics = json.load(file)

# Extract data for DDPG
ddpg_episodes = ddpg_metrics["episodes"]
ddpg_rewards = ddpg_metrics["rewards"]

# Extract data for SAC
sac_episodes = sac_metrics["episodes"]
sac_rewards = sac_metrics["rewards"]

# Define a smoothing function
def smooth_data(data, window_size):
    """
    Smooths the input data using a moving average approach.
    Args:
        data (list): The data to smooth.
        window_size (int): The size of the moving window.
    Returns:
        list: The smoothed data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Set smoothing window size
window_size = 100

# Smooth the rewards data
ddpg_rewards_smoothed = smooth_data(ddpg_rewards, window_size)
sac_rewards_smoothed = smooth_data(sac_rewards, window_size)

# Adjust episodes for the smoothed data
ddpg_episodes_smoothed = ddpg_episodes[:len(ddpg_rewards_smoothed)]
sac_episodes_smoothed = sac_episodes[:len(sac_rewards_smoothed)]

# Plot the smoothed rewards for DDPG and SAC
plt.figure(figsize=(15, 8))
plt.plot(ddpg_episodes_smoothed, ddpg_rewards_smoothed, label="DDPG", alpha=0.8, linestyle=next(line_styles))
plt.plot(sac_episodes_smoothed, sac_rewards_smoothed, label="SAC", alpha=0.8, linestyle=next(line_styles))
plt.title("Rewards vs Episodes: DDPG vs SAC")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.savefig(SAVE_DIR + "compare_sac_ddpg_rewards.png")
