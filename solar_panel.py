import numpy as np

class SolarPanel:
    def __init__(self, capacity=25):
        self.capacity = capacity
        self.current_generation = 0

    def get_generation(self, time_step, max_time_steps):
        # Simulate solar generation based on time of day
        normalized_time = time_step / max_time_steps
        if 0.25 <= normalized_time <= 0.75:  # Peak solar generation between 25% and 75% of the day
            self.current_generation = self.capacity * np.sin(np.pi * (normalized_time - 0.25))
        else:
            self.current_generation = 0  # No generation at night
        return max(0, self.current_generation)
