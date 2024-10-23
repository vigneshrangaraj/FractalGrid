import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium import spaces

# Detect if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")


class Actor(nn.Module):
    def __init__(self, input_size, continuous_action_size, discrete_action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Output layers
        self.continuous_action_layer = nn.Linear(256, continuous_action_size)  # Continuous actions
        self.discrete_action_layer = nn.Linear(256, discrete_action_size)  # Discrete actions

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # Continuous actions (bounded between [-1, 1])
        continuous_action = torch.tanh(self.continuous_action_layer(x))

        # Discrete actions (using softmax to get probabilities)
        discrete_action_probs = torch.softmax(self.discrete_action_layer(x), dim=-1)

        return continuous_action, discrete_action_probs

    def sample_action(self, state):
        # Forward pass to get actions
        continuous_action, discrete_action_probs = self.forward(state)

        # Sample discrete action from the probability distribution
        discrete_action_dist = torch.distributions.Categorical(discrete_action_probs)
        discrete_action = discrete_action_dist.sample()

        return continuous_action, discrete_action




class Critic(nn.Module):
    def __init__(self, input_size, continuous_action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size + continuous_action_size, 256)  # State and action as input
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # Output a single Q-value

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ActorCriticAgent:
    def __init__(self, observation_space, action_space, gamma=0.99, tau=0.001, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, actor_lr=0.0001, critic_lr=0.001,
                 batch_size=128, memory_size=10000):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        input_size = np.prod(self.observation_space.shape)
        self.continuous_action_size = action_space[0].shape[0]
        self.discrete_action_size = action_space[1].n

        self.epsilon = epsilon

        # Initialize actor and critic networks
        self.actor = Actor(input_size, self.continuous_action_size, self.discrete_action_size).to(device)
        self.critic = Critic(input_size, self.continuous_action_size).to(device)

        # Initialize target networks
        self.target_actor = Actor(input_size, self.continuous_action_size, self.discrete_action_size).to(device)
        self.target_critic = Critic(input_size, self.continuous_action_size).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Memory to store transitions
        self.memory = deque(maxlen=memory_size)

        # Initialize target networks to match the main networks
        self.update_target_models()

    def update_target_models(self):
        """
        Initialize target models to have the same weights as the current models.
        """
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update(self, source, target):
        """
        Soft update of target network parameters.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, test_mode=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Get continuous and discrete action outputs from the actor
        continuous_action, discrete_action_probs = self.actor(state_tensor)

        # Convert actions to numpy
        continuous_action = continuous_action.cpu().detach().numpy()[0]

        # Handle exploration during training
        if not test_mode and np.random.rand() <= self.epsilon:  # Exploration during training
            continuous_action += np.random.normal(0, 0.1, self.continuous_action_size)
            # Sample discrete action from the probabilities (with exploration)
            discrete_action = np.random.randint(2, size=self.discrete_action_size)
        else:
            # Exploit during testing
            discrete_action = np.round(discrete_action_probs.cpu().detach().numpy()[0])

        # Clip the continuousact
        #
        # action to be within bounds
        continuous_action = np.tanh(continuous_action)

        # Decay epsilon after each action during training
        if not test_mode and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return continuous_action, discrete_action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from the memory
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device).float()
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device).float()
            action_tensor = torch.FloatTensor(action[0]).unsqueeze(0).to(device).float()  # Continuous action
            discrete_action = action[1]  # Discrete action is already scalar

            reward_tensor = torch.FloatTensor([reward]).to(device).float()
            done_tensor = torch.FloatTensor([done]).to(device).float()

            # Update Critic
            next_continuous_action, _ = self.target_actor(next_state_tensor)
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * self.target_critic(next_state_tensor,
                                                                                           next_continuous_action).detach()
            current_q = self.critic(state_tensor, action_tensor)
            critic_loss = nn.MSELoss()(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update Actor (for continuous actions)
            predicted_continuous_action, predicted_discrete_action_probs = self.actor(state_tensor)
            actor_loss_continuous = -self.critic(state_tensor,
                                                 predicted_continuous_action).mean()  # Maximize Q-value for continuous action

            # Discrete action policy loss
            discrete_action_tensor = torch.LongTensor(np.array(discrete_action)).to(device).float()
            actor_loss_discrete = nn.BCELoss()(predicted_discrete_action_probs.squeeze(0), discrete_action_tensor)

            # Combine the losses (continuous and discrete)
            actor_loss = actor_loss_continuous + actor_loss_discrete

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Soft update of target networks
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def load(self, name):
        """
        Load the model from file.
        """
        self.actor.load_state_dict(torch.load(name))
        self.actor.eval()

    def save(self, name):
        """
        Save the model to file.
        """
        torch.save(self.actor.state_dict(), name)
