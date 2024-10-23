MAX_MG_POWER = 100  # kW, maximum power generation
# Solar Panel Configuration
SOLAR_CAPACITY = 10  # kW, maximum power generation for solar panels
SOLAR_EFFICIENCY = 0.2  # Efficiency of solar panel (optional, depending on detailed modeling)

# Load Configuration
LOAD_BASE_DEMAND = 5  # kW, base load demand for each home
LOAD_VARIANCE = 1.5  # kW, potential variance in load demand
MAX_LOAD = 100  # kW, maximum load

# Energy Storage System (ESS) Configuration
ESS_MAX_CAPACITY = 200 # kWh, maximum energy storage capacity
ESS_INITIAL_SOC = 50  # % (state of charge), initial SOC for ESS
ESS_CHARGE_EFFICIENCY = 0.9  # Charge efficiency (as a fraction)
ESS_DISCHARGE_EFFICIENCY = 0.9  # Discharge efficiency (as a fraction)
ESS_MAX_CHARGE_POWER = 50  # kW, maximum charging power
ESS_MAX_DISCHARGE_POWER = 50  # kW, maximum discharging power
ESS_MIN_SOC = 15  # Minimum SOC allowed to prevent over-discharging
ESS_MAX_SOC = 100  # Maximum SOC allowed to prevent over-charging

# PV System Configuration
PV_MAINTANENCE_COST = 0.05  # $/kWh, maintenance cost of PV system
PV_INVERTER_EFFICIENCY = 0.9  # Inverter efficiency (as a fraction)
PV_DEGRADATION_FACTOR = 0.01  # Degradation factor (as a fraction)
PV_DEGRADATION_RATE = 0.01  # Rate of degradation (as a fraction per time step)

# Main Grid Configuration
GRID_BUY_PRICE = 0.15  # $/kWh, price to buy electricity from the grid
GRID_SELL_PRICE = 0.10  # $/kWh, price to sell electricity back to the grid

# Microgrid & Switch Configuration
NUM_MICROGRIDS = 3  # Number of interconnected microgrids
SWITCH_STATES = [0, 1]  # 0: Open (no energy flow), 1: Closed (energy flow allowed)
MAX_POWER_TRANSFER = 50  # kW, maximum power that can be transferred between microgrids

# Time Step & Simulation Settings
TIME_STEP_DURATION = 1  # 1 hour per time step in the simulation
EPISODES = 1000  # Number of episodes for training the reinforcement learning agent
MAX_STEPS_PER_EPISODE = 24  # Maximum number of time steps per episode (e.g., 24 hours)
DISCOUNT_FACTOR = 0.99  # Discount factor (gamma) for future rewards in RL

# Reward Calculation Weights (optional, to control impact of different factors on reward)
REWARD_ENERGY_BALANCE_WEIGHT = 1.0  # Weight for maintaining energy balance
REWARD_COST_WEIGHT = 0.5  # Weight for minimizing operational costs
REWARD_SWITCH_PENALTY = -0.1  # Penalty for frequent switch toggling (to avoid instability)

# Epsilon-Greedy Exploration
EPSILON_START = 1.0  # Starting value of epsilon (fully random actions at the beginning)
EPSILON_END = 0.1  # Final value of epsilon (more exploitation after learning)
EPSILON_DECAY = 0.995  # Epsilon decay factor after each episode (decay towards EPSILON_END)

# Learning Rate for Neural Network Optimizer
LEARNING_RATE = 0.001  # Learning rate for training the neural network in the DRL algorithm

# Memory Replay Settings
MEMORY_SIZE = 10000  # Size of the replay memory buffer (how many experiences to store)
BATCH_SIZE = 64  # Size of the minibatch used for training (number of experiences per update)

# Target Network Update
TARGET_UPDATE_FREQUENCY = 100  # Number of steps before updating the target network in DQN

# Neural Network Architecture (used in the DRL agent)
NN_HIDDEN_LAYERS = [128, 128]  # List of hidden layer sizes for the neural network

# Gradient Clipping
MAX_GRAD_NORM = 10.0  # Maximum gradient norm for gradient clipping (to stabilize training)

# Discount Factor for Future Rewards
GAMMA = 0.99  # Discount factor for future rewards (same as DISCOUNT_FACTOR)

# Adam Optimizer Parameters
ADAM_BETA1 = 0.9  # Beta1 hyperparameter for the Adam optimizer
ADAM_BETA2 = 0.999  # Beta2 hyperparameter for the Adam optimizer
ADAM_EPSILON = 1e-08  # Epsilon hyperparameter for the Adam optimizer
