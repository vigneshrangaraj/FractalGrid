import json
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

SAVE_DIR = "save/"

plt.rcParams.update({
    'font.size': 20,  # Set default font size
    'axes.titlesize': 20,  # Title font size
    'axes.titleweight': 'bold',  # Title font weight
    'axes.labelsize': 20,  # Axis label font size
    'axes.labelweight': 'bold',  # Axis label font weight
    'xtick.labelsize': 20,  # X-axis tick label font size
    'ytick.labelsize': 20,  # Y-axis tick label font size
    'legend.fontsize': 20,  # Legend font size
    'lines.linewidth': 2,  # Default line thickness
    'axes.grid': False  # Disable grid lines
})

# Define line styles for black-and-white compatibility
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

# File paths and labels
file_paths = [
    "save/metrics_sac_128_p99_p1_256.json",
    "save/metrics_sac_256_p95_p1_256.json",
    "save/metrics_sac_256_p99_p1_128.json",
    "save/metrics_sac_256_p99_p1_256.json",
    "save/metrics_sac_256_p99_p1_512.json",
    "save/metrics_sac_256_p99_p2_256.json",
]
labels = [
    "Run 1",
    "Run 2",
    "Run 3",
    "Run 4",
    "Run 5",
    "Run 6",
]


# Smoothing function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# Initialize plot
plt.figure(figsize=(15, 8))

# Load, smooth, and plot data
window_size = 100
for file_path, label in zip(file_paths, labels):
    with open(file_path, 'r') as f:
        data = json.load(f)
        episodes = data.get("episodes", [])
        rewards = data.get("rewards", [])

        if len(rewards) >= window_size:
            smoothed_rewards = moving_average(rewards, window_size)
            smoothed_episodes = episodes[:len(smoothed_rewards)]
            plt.plot(smoothed_episodes, smoothed_rewards, label=label, linestyle=next(line_styles))
        else:
            print(f"Skipping {file_path}: Not enough data for the specified window size.")

# Add plot details
plt.title("Smoothed Reward Comparison Over Episodes (Window Size = 100)")
plt.xlabel("Episodes")
plt.ylabel("Smoothed Rewards")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt.savefig(SAVE_DIR + "reward_comparison.png", dpi=300)
