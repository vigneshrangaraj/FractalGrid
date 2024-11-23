import numpy as np

class EnergyStorageSystem:
    def __init__(self, capacity, soc_min, soc_max, charge_max, discharge_max, efficiency=0.80):
        self.capacity = capacity  # ESS capacity in kWh
        self.soc_min = soc_min  # Minimum state of charge
        self.soc_max = soc_max  # Maximum state of charge
        self.charge_max = charge_max  # Max charging power in kW
        self.discharge_max = discharge_max  # Max discharging power in kW
        self.efficiency = efficiency  # Efficiency of charging/discharging
        self.soc = np.random.uniform(soc_min, soc_max)  # Initialize SOC between min and max

    def get_soc(self):
        """Return current state of charge (SOC)."""
        return self.soc

    def get_available_battery_power(self):
        """
        Calculate the available power for charging and discharging.

        Returns:
        - available_charge_power: The maximum power that can be absorbed for charging based on SOC.
        - available_discharge_power: The maximum power that can be provided for discharging based on SOC.
        """
        # Available power for charging
        available_capacity = self.capacity * (self.soc_max - self.soc) / 100 # Remaining capacity in kWh
        available_charge_power = min(self.charge_max,
                                     available_capacity)  # Limited by charge_max and remaining capacity

        # Available power for discharging
        available_energy = self.capacity * (self.soc - self.soc_min) / 100  # Available energy in kWh
        available_discharge_power = min(self.discharge_max,
                                        available_energy)  # Limited by discharge_max and available energy

        return available_charge_power, available_discharge_power

    def is_daytime(self, time_step):
        # Assuming a 24-hour cycle where time_step 0-11 is nighttime, 12-23 is daytime
        return 6 <= time_step <= 18

    def charge(self, power, time_interval, time_step):
        """Charge the battery with a given power over a specific time interval."""
        if not self.is_daytime(time_step):
            return
        # Ensure power is within charging limit
        power = min(power, self.charge_max)
        # Compute maximum allowable power based on remaining capacity
        max_power_possible = (self.capacity * (self.soc_max - self.soc))
        power = min(power, max_power_possible)
        # Calculate the power on the DC side
        power_dc = power * self.efficiency
        # Update SOC
        self.soc += (power_dc) / self.capacity
        # Ensure SOC remains within bounds
        self.soc = min(self.soc, self.soc_max)

    def discharge(self, power, time_interval):
        """Discharge the battery with a given power over a specific time interval."""
        # Ensure power is within discharging limit
        power = min(power, self.discharge_max)
        # Compute maximum allowable power based on remaining capacity
        max_power_possible = (self.capacity * (self.soc - self.soc_min))
        power = min(power, max_power_possible)
        # Calculate the power on the DC side
        power_dc = power / self.efficiency
        # Update SOC
        self.soc -= power_dc / self.capacity
        # Ensure SOC remains within bounds
        self.soc = max(self.soc, self.soc_min)

    def can_charge(self):
        """Check if the battery can be charged based on SOC."""
        return self.soc < self.soc_max

    def can_discharge(self):
        """Check if the battery can be discharged based on SOC."""
        return self.soc > self.soc_min
