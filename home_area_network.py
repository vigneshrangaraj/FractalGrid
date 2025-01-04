import numpy as np
import random

class HomeAreaNetwork:
    def __init__(self, base_load=5, load_variance=20):
        """
        :param solar_panel: An object representing the solar panel system.
        :param base_load: The constant base load of the home in kW.
        :param load_variance: The maximum variance in consumer demand in kW.
        """
        self.base_load = base_load  # Constant base load
        self.load_variance = load_variance  # Variance in consumer demand

    def get_demand(self, time_step, max_time_steps):
        """
        Generalized duck curve model with smoother transitions.

        Parameters:
            time_step: Current time step in the day.
            max_time_steps: Total time steps in the simulation.

        Returns:
            Net load values (kW).
        """
        # Base demand curve (sinusoidal pattern)
        demand = self.base_load + self.load_variance * np.sin((2 * np.pi * time_step) / max_time_steps)

        # Midday dip (Gaussian curve)
        midday_dip = 20 * np.exp(-((time_step - max_time_steps / 2)**2) / (2 * (max_time_steps / 12)**2))

        # Evening ramp-up (smooth sigmoid curve)
        evening_ramp = 20 / (1 + np.exp(-0.5 * (time_step - 0.75 * max_time_steps)))

        # Net load
        net_load = demand - midday_dip + evening_ramp

        return max(self.base_load, net_load) + random.randint(0, self.load_variance)

    def set_current_load(self, load):
        """
        Set the current load manually (for testing or override purposes).
        :param load: The load to be set.
        """
        self.current_load = load
