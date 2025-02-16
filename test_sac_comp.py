import json
import numpy as np
from sac import SAC  # Assuming SAC is your model implementation
from environment import YourEnvironment  # Replace with your environment class

# Hyperparameter configurations
configurations = [
    {"batch_size": 256, "gamma": 0.99, "alpha": 0.1, "hidden_neurons": 256},  # Run 1
    {"batch_size": 128, "gamma": 0.99, "alpha": 0.1, "hidden_neurons": 256},  # Run 2
    {"batch_size": 256, "gamma": 0.95, "alpha": 0.1, "hidden_neurons": 256},  # Run 3
    {"batch_size": 256, "gamma": 0.99, "alpha": 0.2, "hidden_neurons": 256},  # Run 4
    {"batch_size": 256, "gamma": 0.99, "alpha": 0.1, "hidden_neurons": 128},  # Run 5
    {"batch_size": 256, "gamma": 0.99, "alpha": 0.1, "hidden_neurons": 512},  # Run 6
]


# Function to save results as JSON
def save_results(run_id, daily_costs, days):
    results = {"day": days, "daily_costs": daily_costs}
    file_name = f"run_{run_id}.json"
    with open(file_name, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {file_name}")


# Test each configuration
def test_sac():
    for run_id, config in enumerate(configurations, start=1):
        print(f"Testing Run {run_id} with config: {config}")

        # Initialize environment and SAC model
        env = YourEnvironment()  # Replace with your environment initialization
        sac_agent = SAC(
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            alpha=config["alpha"],
            hidden_neurons=config["hidden_neurons"]
        )

        # Run the simulation
        num_days = 100  # Example: Test for 100 days
        daily_costs = []
        days = []

        for day in range(1, num_days + 1):
            state = env.reset()  # Reset environment at the start of each day
            done = False
            total_cost = 0

            while not done:
                action = sac_agent.select_action(state)  # Select action
                next_state, reward, done, _ = env.step(action)  # Step through environment
                total_cost += reward
                state = next_state

            daily_costs.append(total_cost)
            days.append(day)

        # Save results for this run
        save_results(run_id, daily_costs, days)


if __name__ == "__main__":
    test_sac()