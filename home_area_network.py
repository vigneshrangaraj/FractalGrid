import numpy as np

class HomeAreaNetwork:
    def __init__(self, base_load=20, load_variance=40):
        self.base_load = base_load
        self.load_variance = load_variance
        self.current_load = base_load

    def get_demand(self, time_step, max_time_steps):
        # Normalize time to range [0, 1]
        normalized_time = time_step / max_time_steps

        # Use a sine wave to simulate load with peak during the day
        # Peak around 0.6 (early evening), minimum around 0.1 (late night)
        peak_amplitude = self.load_variance
        load_fluctuation = peak_amplitude * np.sin(2 * np.pi * (normalized_time - 0.1))

        # Base load is constant, add the deterministic fluctuation
        return abs(self.base_load + load_fluctuation)

    def set_current_load(self, load):
        self.current_load = load
