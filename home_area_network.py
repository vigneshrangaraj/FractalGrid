import numpy as np

class HomeAreaNetwork:
    def __init__(self, base_load=20, load_variance=20):
        self.base_load = base_load
        self.load_variance = load_variance
        self.current_load = base_load

    def get_demand(self, time_step, max_time_steps):
        """
        Get the energy demand for the current time step with added random fluctuations.
        :param time_step: The current time step in the simulation.
        :param max_time_steps: The total number of time steps in a day (e.g., 24 for 24 hours).
        :return: The energy demand with randomness.
        """
        # Normalize time to range [0, 1]
        normalized_time = time_step / max_time_steps

        # Use a sine wave to simulate load with peak during the day
        # Peak around 0.6 (early evening), minimum around 0.1 (late night)
        peak_amplitude = self.load_variance
        deterministic_fluctuation = peak_amplitude * np.sin(2 * np.pi * (normalized_time - 0.1))

        # Introduce random noise with a Gaussian distribution
        noise_amplitude = peak_amplitude * 0.3  # Adjust this value to control the noise level
        random_fluctuation = np.random.normal(0, noise_amplitude)

        # Base load is constant, add deterministic and random fluctuations
        demand = abs(self.base_load + deterministic_fluctuation + random_fluctuation)

        return demand

    def set_current_load(self, load):
        self.current_load = load
