import numpy as np
import matplotlib.pyplot as plt
from sympy.benchmarks.bench_meijerint import alpha

from sac_dqn_agent import SoftActorCriticAgent
from fractal_grid import FractalGrid  # Import your FractalGrid environment
import torch
import json

# Parameters
EPISODES = 15000  # Number of training episodes
MAX_STEPS = 100  # Max steps per episode
TARGET_UPDATE = 10  # How often to update the target model
SAVE_MODEL = True  # Save model after training
MODEL_NAME_ACTOR = "sac_fractal_grid_actor.pth"  # Name of the model file
MODEL_NAME_CRITIC1 = "sac_fractal_grid_critic1.pth"
MODEL_NAME_CRITIC2 = "sac_fractal_grid_critic2.pth"

SMOOTHING_WINDOW = 100  # Size of the moving average window for smoothing rewards
SAVE_DIR = "save/"

METRICS_FILE = SAVE_DIR + "metrics_sac.json"  # File to save metrics

# Detect if MPS is available
device = torch.device("cpu")

print(f"Using device: {device}")

# Initialize the environment
env = FractalGrid(num_microgrids=1)

# Initialize DQN agent
observation_space = env.observation_space
action_space = env.action_space
agent = SoftActorCriticAgent(state_size=env.observation_space.shape[0],  # Assuming observation space is a flat vector
    action_size=action_space.shape[0])  # Number of continuous actions


# Analyze action exploration
def analyze_action_exploration(action_logs, num_bins=20):
    """
    Visualize the distribution of actions across episodes.

    :param action_logs: Array of logged actions from training.
    :param num_bins: Number of bins for the histogram.
    """
    # # Flatten the action logs for the histogram
    # if action_logs.ndim > 2:
    #     flat_actions = action_logs.reshape(-1, action_logs.shape[-1])  # For multidimensional actions
    # else:
    #     flat_actions = action_logs.flatten()  # For single-dimensional actions

    # plt.figure(figsize=(12, 6))
    # plt.hist(flat_actions, bins=num_bins, alpha=0.7, color=['b', 'r', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'], edgecolor="black")
    # plt.title("Action Distribution Across All Episodes")
    # plt.xlabel("Action Values")
    # plt.ylabel("Frequency")
    # plt.grid()
    # plt.savefig(SAVE_DIR + "action_distribution.png")
    # plt.close()

    # Convert action_logs to a NumPy array for easier indexing
    # action_logs = np.array(action_logs)
    # num_action_types = action_logs.shape[-1]  # Number of action dimensions

    # Define consistent colors for each grouping
    # colors = {
    #     "Charge/Discharge": "blue",
    #     "Buy/Sell": "green",
    #     "PV Dispatch": "orange",
    # }
    #
    # plt.figure(figsize=(12, 6))
    #
    # # Plot mean actions for each dimension
    # for i in range(num_action_types):
    #     mean_actions = [np.mean(ep[:, i]) for ep in action_logs if len(ep) > 0]
    #
    #     # Determine the action type (group by Charge/Discharge, Buy/Sell, PV Dispatch)
    #     action_type = ["Charge/Discharge", "Buy/Sell", "PV Dispatch"][i % 3]
    #
    #     # Use consistent color for the action type
    #     plt.plot(
    #         mean_actions,
    #         label=f"Action {i + 1}: {action_type}",
    #         color=colors[action_type]
    #     )
    #
    # plt.title("Mean Actions Taken Per Episode")
    # plt.xlabel("Episode")
    # plt.ylabel("Mean Action Value")

    # Position the legend to the right side
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    # plt.grid()
    #
    # # Adjust layout to make space for the legend
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    #
    # # Save and show the plot
    # plt.savefig(SAVE_DIR + "action_diversity_grouped_colors.png", bbox_inches="tight")
    # plt.close()

# Plotting metrics: reward vs. episode, epsilon vs. episode
def plot_metrics():
    assert len(rewards_list) == len(episodes_list), (
        f"Length mismatch: rewards_list({len(rewards_list)}) vs. episodes_list({len(episodes_list)})"
    )

    # Smooth rewards only if there are enough entries
    if len(rewards_list) >= SMOOTHING_WINDOW:
        # Smooth the rewards using a moving average
        smoothed_rewards = np.convolve(rewards_list, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode='valid')
        smoothed_episodes = episodes_list[:len(smoothed_rewards)]  # Adjust episodes to match smoothed rewards
    else:
        # If rewards_list is too short, skip smoothing and use raw values
        smoothed_rewards = rewards_list
        smoothed_episodes = episodes_list

    # Plot smoothed rewards over episodes
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_episodes, smoothed_rewards, label='Rewards', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Rewards vs. Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/sac_rewards_vs_episodes.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(entropies, label="Policy Entropy", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Entropy")
    plt.title("Policy Entropy vs. Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/entropy_vs_episodes.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(alpha_values, label="Alpha (Entropy Coefficient)", color="orange")
    plt.xlabel("Episodes")
    plt.ylabel("Alpha")
    plt.title("Entropy Coefficient vs. Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/alpha_vs_episodes.png")
    plt.close()

# Tracking metrics
rewards_list = []
entropies = []  # To track entropy over episodes
episodes_list = []
alpha_values = []

# Initialize action_logs as a list
action_logs = []

best_reward = -np.inf
update_frequency = 10
log_frequency = 10
# Training loop
for episode in range(EPISODES):
    state = env.reset()
    episode_actions = []
    total_reward = 0
    episode_reward = 0.0
    info = {}
    for step in range(MAX_STEPS):
        # Choose an action using the DQN agent
        action = agent.select_action(state)

        episode_actions.append(action)

        # Take the action and get the result
        next_state, reward, done, _ = env.step(action)

        # Store the transition in memory
        agent.store_transition(state, action, reward, next_state, done)

        agent.update()

        # Accumulate rewards
        episode_reward += reward

        # Move to the next state
        state = next_state

        info = _

        if done:
            break

    # Compute entropy for the current policy
    with torch.no_grad():
        _, log_std = agent.actor(torch.FloatTensor(state).unsqueeze(0).to(device))
        entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum().item()
        entropies.append(entropy)

    alpha_values.append(agent.log_alpha.exp().item())

    if (episode + 1) % log_frequency == 0:
        print(
            f"Episode {episode + 1}/{EPISODES}, Total Reward: {episode_reward}, entropy: {entropy:.2f}"
             ", alpha: {agent.log_alpha.exp().item():.2f" + f", best reward: {best_reward}")



    rewards_list.append(episode_reward)
    episodes_list.append(episode)

    # Append the episode's actions to action_logs
    action_logs.append(episode_actions)

    # Convert action_logs to a NumPy array for analysis and visualization
    action_logs_array = np.array(action_logs, dtype=object)  # Use dtype=object for variable-length sequences

    # Analyze the actions for exploration
    analyze_action_exploration(action_logs_array)

    # Save metrics to file
    metrics = {
        "episodes": episodes_list,
        "rewards": rewards_list,
        "entropies": entropies,
        "alpha": alpha_values
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    # Call the plotting function to save plots
    plot_metrics()

    # log and print the best model
    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save(SAVE_DIR + MODEL_NAME_ACTOR, SAVE_DIR + MODEL_NAME_CRITIC1, SAVE_DIR + MODEL_NAME_CRITIC2)
        print(f"Best model saved at episode {episode + 1} with total reward {episode_reward}")
        print(info)

    # Save every 100 episodes
    if (episode + 1) % 100 == 0:
        agent.save(SAVE_DIR + MODEL_NAME_ACTOR, SAVE_DIR + MODEL_NAME_CRITIC1, SAVE_DIR + MODEL_NAME_CRITIC2)


# Save the model
if SAVE_MODEL:
    agent.save(SAVE_DIR + MODEL_NAME_ACTOR, SAVE_DIR + MODEL_NAME_CRITIC1, SAVE_DIR + MODEL_NAME_CRITIC2)
