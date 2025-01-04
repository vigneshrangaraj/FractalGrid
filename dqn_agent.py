import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Detect if MPS is available
device = torch.device("cpu")

print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_size, continuous_action_size, hidden_size=256, action_low=-1.0, action_high=1.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # Use LayerNorm instead of BatchNorm
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)  # Use LayerNorm instead of BatchNorm
        self.continuous_output = nn.Linear(hidden_size, continuous_action_size)
        self.tanh = nn.Tanh()  # For continuous actions between [-1, 1]
        self.sigmoid = nn.Sigmoid()  # For binary actions

        # Define action bounds
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))  # LayerNorm instead of BatchNorm
        x = torch.relu(self.ln2(self.fc2(x)))  # LayerNorm instead of BatchNorm
        raw_action = self.continuous_output(x)

        # Apply tanh to bound the actions to [-1, 1]
        tanh_action = self.tanh(raw_action)

        # Scale the tanh output to the desired action range [action_low, action_high]
        scaled_action = self.action_low + (self.action_high - self.action_low) * (tanh_action + 1) / 2
        return scaled_action


class Critic(nn.Module):
    def __init__(self, state_size, continuous_action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + continuous_action_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # Use LayerNorm instead of BatchNorm
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)  # Use LayerNorm instead of BatchNorm
        self.fc3 = nn.Linear(hidden_size, 1)  # Output Q-value

    def forward(self, state, continuous_action):
        # Concatenate state with continuous and discrete actions
        x = torch.cat([state, continuous_action], dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))  # LayerNorm instead of BatchNorm
        x = torch.relu(self.ln2(self.fc2(x)))  # LayerNorm instead of BatchNorm
        Q_value = self.fc3(x)
        return Q_value



class ActorCriticAgent:
    def __init__(self, state_size, continuous_action_size, gamma=0.99, tau=0.001,
                 learning_rate=0.0001, batch_size=64, memory_size=10000, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99995):
        self.state_size = state_size
        self.continuous_action_size = continuous_action_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon  # Epsilon for discrete actions
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # Actor networks (for continuous and discrete actions)
        self.actor = Actor(state_size, continuous_action_size).to(device)
        self.actor_target = Actor(state_size, continuous_action_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # Critic networks
        self.critic = Critic(state_size, continuous_action_size).to(device)
        self.critic_target = Critic(state_size, continuous_action_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Noise process for continuous actions (exploration)
        self.noise = np.zeros(continuous_action_size)
        self.noise_theta = 0.15
        self.noise_sigma = 0.2

    def add_noise(self, action):
        # Ornstein-Uhlenbeck noise for continuous actions
        self.noise = self.noise + self.noise_theta * (-self.noise) + self.noise_sigma * np.random.randn(
            self.continuous_action_size)
        return action + self.noise

    def select_action(self, state, test_mode=False, add_noise=True):
        """
        Selects actions for both continuous and discrete spaces based on the current policy (actor).
        - In training mode: Continuous actions use noise, and discrete actions use epsilon-greedy exploration.
        - In test mode: Both continuous and discrete actions are deterministic (no exploration).

        :param state: The current state of the environment.
        :param test_mode: If True, the agent will only exploit (no noise or random actions).
        :param add_noise: Whether to add exploration noise to the continuous actions (True during training).
        :return: Tuple of (continuous actions, discrete actions).
        """
        # Convert state to a PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Get continuous and discrete actions from the actor network
        continuous_action = self.actor(state_tensor)

        # Detach actions from the computational graph and convert them to NumPy arrays
        continuous_action = continuous_action.cpu().detach().numpy()[0]

        if test_mode:
            # In test mode: No noise and no epsilon-greedy exploration
            continuous_action = np.clip(continuous_action, 0, 1)  # Exploit continuous actions
        else:
            # Training mode: Epsilon-greedy exploration for both continuous and discrete actions
            if np.random.rand() <= self.epsilon:
                # Random exploration for both continuous and discrete actions
                continuous_action = np.random.uniform(low=-1, high=1,
                                                      size=self.continuous_action_size)  # Random continuous action
            else:
                # Add noise to the continuous actions for exploration
                if add_noise:
                    continuous_action = self.add_noise(continuous_action)

                # Clip continuous actions to valid range [0, 1]
                continuous_action = np.clip(continuous_action, -1, 1)

            # Decay epsilon (for exploration-exploitation trade-off during training)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return continuous_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            # Critic loss calculation (as usual)
            next_continuous_action = self.actor_target(
                torch.FloatTensor(next_state).unsqueeze(0).to(device))
            next_Q = self.critic_target(torch.FloatTensor(next_state).unsqueeze(0).to(device), next_continuous_action)
            target_Q = reward + (1 - done) * self.gamma * next_Q
            current_Q = self.critic(torch.FloatTensor(state).unsqueeze(0).to(device),
                                    torch.FloatTensor(action).unsqueeze(0).to(device))

            # Critic optimization
            critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            # Apply gradient clipping to the critic
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # Clipping gradients
            self.critic_optimizer.step()

            # Actor loss calculation
            predicted_continuous_action = self.actor(
                torch.FloatTensor(state).unsqueeze(0).to(device))
            actor_loss = -self.critic(torch.FloatTensor(state).unsqueeze(0).to(device), predicted_continuous_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # Apply gradient clipping to the actor
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # Clipping gradients
            self.actor_optimizer.step()

        # Soft update for target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        # Decay epsilon (if using for discrete actions)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor.eval()
        self.critic.eval()
