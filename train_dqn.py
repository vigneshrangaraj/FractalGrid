import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import ActorCriticAgent
from fractal_grid import FractalGrid  # Import your FractalGrid environment
import torch

# Parameters
EPISODES = 15000  # Number of training episodes
MAX_STEPS = 100  # Max steps per episode
TARGET_UPDATE = 10  # How often to update the target model
SAVE_MODEL = True  # Save model after training
MODEL_NAME_ACTOR = "dqn_fractal_grid_actor.pth"  # Name of the model file
MODEL_NAME_CRITIC = "dqn_fractal_grid_critic.pth"
SAVE_DIR = "save/"

# Initialize the environment
env = FractalGrid(num_microgrids=7)

# Initialize DQN agent
observation_space = env.observation_space
action_space = env.action_space
agent = ActorCriticAgent(state_size=env.observation_space.shape[0],  # Assuming observation space is a flat vector
    continuous_action_size=action_space[0].shape[0],  # Number of continuous actions
    discrete_action_size=action_space[1].n)  # Number of discrete actions (e.g., binary switches))

# Tracking metrics
rewards_list = []
epsilon_list = []
episodes_list = []

best_reward = -np.inf
update_frequency = 10
log_frequency = 10
# Training loop
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    episode_reward = 0.0
    info = {}
    for step in range(MAX_STEPS):
        # Choose an action using the DQN agent
        action = agent.select_action(state)

        # Take the action and get the result
        next_state, reward, done, _ = env.step(action)

        # Store the transition in memory
        agent.remember(state, action, reward, next_state, done)

        if episode % update_frequency == 0:
            agent.replay()

        # Accumulate rewards
        episode_reward += reward

        # Move to the next state
        state = next_state

        info = _

        if done:
            break
    if (episode + 1) % log_frequency == 0:
        print(
            f"Episode {episode + 1}/{EPISODES}, Total Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f} best reward: {best_reward}")
    # Decay epsilon after each episode
    epsilon_list.append(agent.epsilon)
    rewards_list.append(episode_reward)
    episodes_list.append(episode)

    # log and print the best model
    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save(SAVE_DIR + MODEL_NAME_ACTOR, SAVE_DIR + MODEL_NAME_CRITIC)
        print(f"Best model saved at episode {episode + 1} with total reward {episode_reward}")
        print(info)


# Save the model
if SAVE_MODEL:
    agent.save(SAVE_DIR + MODEL_NAME_ACTOR, SAVE_DIR + MODEL_NAME_CRITIC)


# Plotting metrics: reward vs. episode, epsilon vs. episode
def plot_metrics():
    # Plot rewards over episodes
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, rewards_list, label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Rewards vs. Episodes')
    plt.legend()
    plt.grid()
    plt.savefig(SAVE_DIR + "rewards_vs_episodes.png")

    # Plot epsilon decay
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, epsilon_list, label='Epsilon Decay')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay vs. Episodes')
    plt.legend()
    plt.grid()
    plt.savefig(SAVE_DIR + "epsilon_decay_vs_episodes.png")


# Call the plotting function to save plots
plot_metrics()
