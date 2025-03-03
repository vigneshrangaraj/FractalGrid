from solar_panel import SolarPanel
from ess import EnergyStorageSystem
from home_area_network import HomeAreaNetwork

from utils import config


class MicroGrid:
    def __init__(self, delta_t=1, max_grid_power=25, seed=0):
        self.solar_panel = SolarPanel(max_grid_power)
        self.ess = EnergyStorageSystem(config.ESS_MAX_CAPACITY, config.ESS_MIN_SOC, config.ESS_MAX_SOC, config.ESS_MAX_CHARGE_POWER, config.ESS_MAX_DISCHARGE_POWER)
        self.load = HomeAreaNetwork(seed=seed, use_seed=False)
        self.switch = 0  # Assume switch is open initially
        self.delta_t = delta_t
        self.degradation_factor = 1
        self.dispatch_pv_power = 0
        self.current_demand = 0
        self.max_grid_power = max_grid_power
        self.final_net_power = 20
        self.power_available_from_neighbors = 0

        self.neighbors = []  # List of neighboring microgrids
        self.switches = {}  # Dictionary of switches (bi-directional) to neighbors

        self.power_transfer_t = 0
        self.power_transfer_grid_t = None
        self.power_transfer_switch_t = None

        self.index = 0

    def reset_state(self):
        return [
            0,
            0,
            0,
            0,
            0
        ]


    def get_state(self, time_step):
            return [
                self.solar_panel.get_generation( time_step, 23),
                self.ess.get_soc(),
                self.load.get_demand(time_step, 23),
                self.get_final_net_power(),
                self.get_power_available_from_neighbors()
            ]

    def get_power_available_from_neighbors(self):
        return self.power_available_from_neighbors

    def get_final_net_power(self):
        return self.final_net_power

    def add_neighbor(self, neighbor, switch_name):
        self.neighbors.append(neighbor)
        self.switches[switch_name] = 0  # Initially, all switches are open (0)

    def set_switch(self, switch_name, state):
        """
        Set the state of a switch (0 for open, 1 for closed).

        Parameters:
        - switch_name: The name of the switch.
        - state: 0 (open) or 1 (closed).
        """
        if switch_name in self.switches:
            self.switches[switch_name] = state

    def get_net_power(self, pv_dispatch_power, load_demand):
        battery_power = self.ess.get_available_battery_power()[1]

        net_power = pv_dispatch_power + battery_power - load_demand
        return net_power

    def send_power_to_neighbor(self, power):
        """
        Send power to a neighboring microgrid.
        """
        # Reduce the local net power by the amount of power sent
        self.ess.discharge(power, 1)

    def receive_power_from_neighbor(self, power):
        """
        Receive power from a neighboring microgrid.
        """
        # Increase the local net power by the amount of power received
        self.ess.charge(power, 1)

    def get_power_from_grid(self, time_step):
        # Power generated by solar and stored in the ESS
        solar_generation = self.solar_panel.get_generation(time_step, 23)
        load_demand = self.load.get_demand(time_step, 23)

        # Net power balance: positive if generation exceeds demand, negative if demand exceeds generation
        net_power = solar_generation - load_demand

        # If the net power is negative (demand > generation), buy power from the grid
        if net_power < 0:
            power_from_grid = min(abs(net_power), self.max_grid_power)
        else:
            power_from_grid = 0

        return power_from_grid

    def dispatch_power(self, pv_dispatch_power):
        self.dispatch_pv_power = max(0, pv_dispatch_power)  # Ensure that dispatch_power is non-negativepv_dispatch_power

    def calculate_local_net_power(self, pv_dispatch_power, load_demand):
        if pv_dispatch_power <= 0:
            pv_dispatch_power = 0
        net_power = pv_dispatch_power - load_demand
        return net_power

    def get_power_to_grid(self, time_step):
        # Power generated by solar and stored in the ESS
        solar_generation = self.solar_panel.get_generation(time_step=time_step, max_time_steps=23)
        load_demand = self.load.get_demand()

        # Net power balance: positive if generation exceeds demand, negative if demand exceeds generation
        net_power = solar_generation - load_demand

        # If the net power is positive (generation > demand), sell the surplus to the grid
        if net_power > 0:
            power_to_grid = min(net_power, self.max_grid_power)
        else:
            power_to_grid = 0

        return power_to_grid

    def calculate_operational_cost(self, pv_power_output):
        # Maintenance cost (fixed)
        maintenance_cost = config.PV_MAINTANENCE_COST * self.delta_t

        # Degradation cost (increases over time)
        degradation_cost = (1 - config.PV_DEGRADATION_FACTOR) * pv_power_output * self.delta_t

        # Inverter loss cost (inefficiency in converting DC to AC)
        inverter_loss = (1 - config.PV_INVERTER_EFFICIENCY) * pv_power_output * self.delta_t

        # Update the degradation factor for the next time step
        self.degradation_factor -= config.PV_DEGRADATION_RATE * self.delta_t

        # Total cost is the sum of maintenance, degradation, and inverter loss costs
        total_cost = maintenance_cost + degradation_cost + inverter_loss

        return total_cost