import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

# Detect if MPS is available
device = torch.device("cpu")

print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, action_size)
        self.log_std_layer = nn.Linear(hidden_size, action_size)

        # Clamping bounds for log std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        """
        Forward pass to compute the mean and log standard deviation of the Gaussian policy.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # Avoid instability
        return mean, log_std

    def sample(self, state):
        """
        Samples an action using the reparameterization trick.
        Returns: action, log_prob
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal_dist = torch.distributions.Normal(mean, std)
        z = normal_dist.rsample()  # Reparameterization trick
        action = torch.tanh(z)  # Enforce action bounds to [-1, 1]

        # Calculate log probability
        log_prob = normal_dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)  # Adjust for tanh squashing
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        # Define a single Q-value network
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_out = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """
        Compute a single Q-value for the given state and action.
        Args:
            state (Tensor): The current state.
            action (Tensor): The action taken.
        Returns:
            Tensor: Q-value estimate.
        """
        # Concatenate state and action
        xu = torch.cat([state, action], dim=-1)

        # Forward pass
        x = F.relu(self.fc1(xu))
        x = F.relu(self.fc2(x))
        q = self.q_out(x)

        return q


class SoftActorCriticAgent:
    def __init__(self, state_size, action_size, hidden_size=256, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy temperature

        # Replay memory
        self.memory = deque(maxlen=100000)
        self.batch_size = 64

        # Actor network
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks
        self.critic1 = Critic(state_size, action_size).to(device)
        self.critic2 = Critic(state_size, action_size).to(device)
        self.critic1_target = Critic(state_size, action_size).to(device)
        self.critic2_target = Critic(state_size, action_size).to(device)

        # Synchronize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Entropy temperature (learnable)
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Target entropy (heuristic)
        self.target_entropy = -np.prod((action_size,)).item()

    def select_action(self, state, test_mode=False):
        """
        Selects an action given the current state.

        In training mode, the action is sampled stochastically from the policy's distribution.
        In test mode, the deterministic mean action is returned.

        Args:
            state (array): The current state of the environment.
            test_mode (bool): If True, returns deterministic actions.

        Returns:
            numpy.ndarray: The selected action, shape (action_size,).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)  # Actor outputs mean and log_std
            std = torch.exp(log_std)  # Convert log_std to standard deviation

            if test_mode:
                # Deterministic action: return mean of the distribution
                action = mean
            else:
                # Stochastic action: sample from Gaussian distribution
                action = mean + std * torch.randn_like(mean)

            # Apply tanh to bound the action to [-1, 1]
            action = torch.tanh(action)

        # Remove batch dimension and return as numpy array
        return action.squeeze(0).cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """
                    Updates the actor, critics, and entropy coefficient using a mini-batch from the replay buffer.
                    """
        # Ensure there are enough samples in the replay buffer
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch from the replay buffer
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert mini-batch to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Step 1: Critic Updates
        with torch.no_grad():
            # Sample actions and log probabilities for the next states
            next_actions, next_log_probs = self.actor.sample(next_states)

            q1_next, q2_next = self.critic1_target(next_states, next_actions), self.critic2_target(next_states, next_actions)

            # Compute target Q-values
            min_q_next = torch.min(q1_next, q2_next)

            q_target = rewards + (1 - dones) * self.gamma * (min_q_next - self.alpha * next_log_probs)

        # Compute current Q-values
        q1, q2 = self.critic1(states, actions), self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Update critic networks
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Step 2: Actor Update
        # Sample new actions and log probabilities for the current states
        new_actions, log_probs = self.actor.sample(states)

        # Compute actor loss
        q1_new, q2_new = self.critic1(states, new_actions), self.critic2(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 3: Entropy Coefficient Update (if learnable)
        if self.log_alpha.requires_grad:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Step 4: Soft Update of Target Critics
        self.soft_update(self.critic1_target, self.critic1, self.tau)
        self.soft_update(self.critic2_target, self.critic2, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, actor_path, critic1_path, critic2_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load(self, actor_path, critic1_path, critic2_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic1.load_state_dict(torch.load(critic2_path))
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

