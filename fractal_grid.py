import gymnasium as gym
from gymnasium import spaces
import numpy as np

from main_grid import MainGrid
from microgrid import MicroGrid

from utils import config


class FractalGrid(gym.Env):
    def __init__(self, num_microgrids=3, max_time_steps=23):
        super(FractalGrid, self).__init__()

        # Initialize variables
        self.num_microgrids = num_microgrids
        self.max_power = config.MAX_MG_POWER
        self.max_load = config.MAX_LOAD
        self.max_soc = config.ESS_MAX_SOC
        self.max_time_steps = max_time_steps  # 24-hour period
        self.main_grid = MainGrid(Pmax_g=config.MAX_MG_POWER, LMP_buy=20, LMP_sell_discount=0.1, delta_t=1)
        self.current_time_step = 0

        self.microgrids, self.total_switches = self.initialize_microgrids(num_microgrids, config.MAX_MG_POWER)

        # Observation Space: SOC, Solar Generation, Load, Grid Power Exchange for each microgrid
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.array([0, 0, 0, -np.inf, -np.inf] * num_microgrids),  # Min values for microgrid features
                np.array([0])  # Min value for timestep
            ]),
            high=np.concatenate([
                np.array(
                    [self.max_soc, self.max_power, self.max_load, self.max_power, self.max_power] * num_microgrids),
                np.array([23])  # Max value for timestep
            ]),
            dtype=np.float32
        )

        # Action Space: Charge/Discharge ESS, Buy/Sell Power to/from Grid, Dispatch PV Power, Switching Action
        # Continuous actions: -1 to 1 for charge/discharge, buy/sell, and 0 to 1 for PV dispatch
        action_space_continuous = spaces.Box(
            low=np.array([-1] * num_microgrids * 3),  # Discharge/sell, PV dispatch
            high=np.array([1] * num_microgrids * 3)  # Charge/buy, PV dispatch
        )

        # Combining the continuous action space with the discrete switching space
        self.action_space = action_space_continuous

        # Initialize logs for metrics
        self.soc_log = [[] for _ in range(self.num_microgrids)]  # Log SOC for each microgrid
        self.net_load_log = [[] for _ in range(self.num_microgrids)]  # Log net load for each microgrid
        self.charge_discharge_log = [[] for _ in range(self.num_microgrids)]  # Log charging/discharging power

        self.state = self._get_initial_state() # Get initial state

    def _get_initial_state(self):
        state = []
        for grid in self.microgrids:
            state.extend(grid.get_state(0))

        # Include the current time step in the state
        state.append(self.current_time_step / self.max_time_steps)  # Normalized time
        return np.array(state)


    @staticmethod
    def initialize_microgrids(num_microgrids, max_power):
        # Initialize and connect microgrids based on a Fractal Tree algorithm
        microgrids = []

        # Initialize the set to track unique switches
        switch_set = set()

        # Connect microgrids using a Fractal Tree structure
        if num_microgrids == 1:
            return microgrids, 0  # Only one microgrid, no switches needed

        for i in range(num_microgrids):
            microgrid = MicroGrid(1, max_power)
            microgrids.append(microgrid)
        for i in range(num_microgrids):
            left_child = 2 * i + 1  # Calculate index of left child
            right_child = 2 * i + 2  # Calculate index of right child

            microgrids[i].index = i

            if left_child < num_microgrids:
                # Create a single bi-directional switch (consistent naming based on min/max index)
                switch_name = f"S_{min(i, left_child)}_to_{max(i, left_child)}"
                microgrids[i].add_neighbor(microgrids[left_child], switch_name)
                microgrids[left_child].add_neighbor(microgrids[i], switch_name)

                # Add the switch to the set (only if it hasn't been added before)
                switch_set.add(switch_name)

            if right_child < num_microgrids:
                # Create a single bi-directional switch (consistent naming based on min/max index)
                switch_name = f"S_{min(i, right_child)}_to_{max(i, right_child)}"
                microgrids[i].add_neighbor(microgrids[right_child], switch_name)
                microgrids[right_child].add_neighbor(microgrids[i], switch_name)

                # Add the switch to the set (only if it hasn't been added before)
                switch_set.add(switch_name)

        total_switches = len(switch_set)
        return microgrids, total_switches

    def step(self, action):
        info = {}
        # Separate continuous and discrete actions
        continuous_actions = action

        total_power_bought = 0
        total_power_sold = 0

        total_transaction_cost = 0
        total_operational_cost = 0

        # Global index to keep track of switching actions
        switch_index = 0
        processed_switches = set()  # Track switches that have already been processed

        # Step 1: Process continuous actions and calculate local net power for each microgrid (before neighbor interactions)
        local_net_powers = []  # To store each microgrid's local net power

        for i in range(self.num_microgrids):
            # Extract the continuous actions for this microgrid
            charge_discharge_action = continuous_actions[i * 3]  # Charging/Discharging ESS
            grid_exchange_action = continuous_actions[i * 3 + 1]  # Buy/Sell power from/to the grid
            pv_dispatch_action = continuous_actions[i * 3 + 2]  # Dispatch PV power (0 to 1)

            grid = self.microgrids[i]

            # Get the available solar generation from the SolarPanel class
            solar_generation = grid.solar_panel.get_generation(self.current_time_step, self.max_time_steps)

            # Handle the charge or discharge action on the Battery class
            battery_power = 0

            # Dispatch PV power for the current microgrid
            pv_dispatch_power = min(self.max_power, abs(pv_dispatch_action) * solar_generation)  # pv_dispatch_action * solar_generation
            grid.dispatch_power(pv_dispatch_power)

            # Calculate the load demand for the microgrid from the Load class
            load_demand = grid.load.get_demand(self.current_time_step, self.max_time_steps)
            grid.current_demand = load_demand
            grid.load.set_current_load(load_demand)

            # Step 1a: Calculate local net power for the current microgrid (before neighbor interactions)
            local_net_power = grid.calculate_local_net_power(pv_dispatch_power, load_demand)

            if local_net_power > 0:
                # Check if SOC is below max limit before charging
                if grid.ess.soc < grid.ess.soc_max:
                    available_capacity = (grid.ess.soc_max - grid.ess.soc) * grid.ess.capacity  # Max energy that can be added
                    max_charging_power = min(available_capacity,
                                             config.ESS_MAX_CHARGE_POWER)  # Ensure we don't exceed SOC limit
                    actual_charging_power = min(abs(charge_discharge_action) * config.ESS_MAX_DISCHARGE_POWER, local_net_power, max_charging_power)
                    # Perform the charging
                    grid.ess.charge(abs(actual_charging_power), 1, self.current_time_step)
                    battery_power = -actual_charging_power  # Positive for charging
                    local_net_power -= actual_charging_power
                else:
                    battery_power = 0  # No charging if SOC is at max
            else:
                if grid.ess.soc > grid.ess.soc_min:
                    available_energy = (grid.ess.soc - grid.ess.soc_min) * grid.ess.capacity  # Max energy that can be discharged
                    max_discharging_power = min(available_energy,
                                                config.ESS_MAX_DISCHARGE_POWER)  # Ensure we don't exceed SOC limit
                    actual_discharging_power = min(abs(charge_discharge_action) * config.ESS_MAX_DISCHARGE_POWER, max_discharging_power)

                    # Perform the discharging
                    grid.ess.discharge(actual_discharging_power, 1)
                    battery_power = actual_discharging_power  # Positive for discharging
                    local_net_power += actual_discharging_power
                else:
                    battery_power = 0  # No discharging if SOC is at min

            local_net_powers.append(local_net_power)  # Store the local net power for this microgrid

            info[f"charge_discharge_{i}"] = battery_power
            info[f"soc_{i}"] = grid.ess.get_soc()

        # Step 2: Process neighbor power exchanges based on switching actions
        # For each neighbor, adjust both microgrids' net power simultaneously if the switch is closed
        neighbor_transfers = [0] * self.num_microgrids  # Store power exchanged with neighbors for each microgrid

        for i, microgrid in enumerate(self.microgrids):
            for neighbor in microgrid.neighbors:
                neighbor_index = neighbor.index

                # Generate a consistent switch name to avoid duplicates
                switch_name = f"S_{min(microgrid.index, neighbor.index)}_to_{max(microgrid.index, neighbor.index)}"

                # Only process this switch if it hasn't been processed before
                if switch_name not in processed_switches:
                    # Get switch state using the global switch index
                    switch_index += 1  # Increment global switch index for next neighbor

                    # Update neighbor power availability and dont double count
                    neighbor_transfers[i] += local_net_powers[neighbor_index]

                    # Get net power of the current microgrid and its neighbor
                    net_power_i = local_net_powers[i]  # Current microgrid's local net power
                    net_power_neighbor = local_net_powers[neighbor_index]  # Neighbor's local net power

                    # Transfer power if necessary based on local power status
                    power_transferred = 0
                    if net_power_i < 0 and net_power_neighbor > 0:
                        # Microgrid i needs power, neighbor has excess
                        power_transferred = min(abs(net_power_i), net_power_neighbor)
                        local_net_powers[i] += power_transferred  # Microgrid i receives power
                        net_power_i += power_transferred
                        local_net_powers[neighbor_index] -= power_transferred  # Neighbor gives power
                        net_power_neighbor -= power_transferred
                        if f"power_transfer_{i}" in info:
                            info[f"power_transfer_{i}"] += power_transferred
                        else:
                            info[f"power_transfer_{i}"] = power_transferred
                        if f"power_transfer_{neighbor_index}" in info:
                            info[f"power_transfer_{neighbor_index}"] -= power_transferred
                        else:
                            info[f"power_transfer_{neighbor_index}"] = -power_transferred
                    elif net_power_i > 0 and net_power_neighbor < 0:
                        # Microgrid i has excess power, neighbor needs power
                        power_transferred = min(net_power_i, abs(net_power_neighbor))
                        local_net_powers[i] -= power_transferred  # Microgrid i gives power
                        net_power_i -= power_transferred
                        local_net_powers[neighbor_index] += power_transferred  # Neighbor receives power
                        net_power_neighbor += power_transferred
                        if f"power_transfer_{i}" in info:
                            info[f"power_transfer_{i}"] -= power_transferred
                        else:
                            info[f"power_transfer_{i}"] = -power_transferred
                        if f"power_transfer_{neighbor_index}" in info:
                            info[f"power_transfer_{neighbor_index}"] += power_transferred
                        else:
                            info[f"power_transfer_{neighbor_index}"] = power_transferred

                    if power_transferred != 0:
                        info[switch_name] = 1
                    else:
                        info[switch_name] = 0

                    neighbor_transfers[i] += power_transferred  # Track transfer for microgrid i
                    neighbor_transfers[neighbor_index] -= power_transferred  # Track transfer for the neighbor

                    # Mark this switch as processed
                    processed_switches.add(switch_name)

            microgrid.final_net_power = local_net_powers[i]
            microgrid.power_available_from_neighbors = neighbor_transfers[i]

        # Step 3: Process grid power exchange and final net power calculations
        for i, microgrid in enumerate(self.microgrids):
            pv_dispatch_power = continuous_actions[i * 3 + 2]  # Dispatch PV power (0 to 1)
            grid = self.microgrids[i]

            # Get the local net power calculated earlier
            net_power = local_net_powers[i]

            P_buy = 0
            P_sell = 0

            # Adjust grid power exchange based on the action and the net power
            grid_exchange_action = continuous_actions[i * 3 + 1]  # Buy/Sell power from/to the grid
            if grid_exchange_action > 0:
                if net_power < 0:
                    # Buy power from the grid based on the action and the net power shortfall
                    P_buy = min(grid_exchange_action * self.max_power, abs(net_power), self.max_power)
                    P_sell = 0  # No selling to the grid
            elif grid_exchange_action < 0:
                if net_power > 0:
                    # Sell power to the grid based on the action and the available surplus power
                    P_sell = min(abs(grid_exchange_action) * self.max_power, net_power, self.max_power)
                    P_buy = 0  # No buying from the grid
            else:
                P_buy = 0
                P_sell = 0

            total_power_bought += P_buy
            total_power_sold += P_sell

            # Calculate transaction cost with the main grid
            transaction_cost = self.main_grid.calculate_transaction_cost(P_buy, P_sell)
            total_transaction_cost += transaction_cost

            # Calculate the operational cost for this microgrid
            operational_cost = grid.calculate_operational_cost(
                grid.solar_panel.get_generation(self.current_time_step, self.max_time_steps))

            # penalize by fining if the net power is negative
            if net_power < 0:
                total_operational_cost += abs(net_power) * 10
            #
            # # incentivize if the power transfer was utilized
            # if neighbor_transfers[i] != 0:
            #     total_operational_cost -= abs(neighbor_transfers[i]) * 5

            total_operational_cost += operational_cost

            # Store net power information for each microgrid
            info[f"net_load_{i}"] = grid.current_demand
            info[f"grid_{i}"] = local_net_powers[i]
            info[f"pv_dispatch_{i}"] = max(0, grid.dispatch_pv_power)
            info[f"neighbor_available_power_{i}"] = sum(neighbor.final_net_power for neighbor in grid.neighbors)

        # Total cost is the sum of operational cost and transaction cost
        total_cost = total_operational_cost + total_transaction_cost

        # Reward is the negative of the total cost (minimize cost)
        reward = -total_cost

        # Check if the episode is done (we can base it on time steps)
        self.current_time_step += 1
        done = self.current_time_step >= self.max_time_steps

        # Get the next state and update it
        next_state = self.get_observation(self.current_time_step)
        self.state = next_state  # Update the current state

        info["total_grid_exchange_power"] = total_power_bought - total_power_sold
        info["total_net_load"] = sum(grid.final_net_power for grid in self.microgrids)

        # Return the new state, reward, done status, and any additional info
        return next_state, reward, done, info

    def get_observation(self, time_step):
        """
        Get the current observation (state) of the environment.
        """
        # Gather observations from all microgrids
        observations = []
        for microgrid in self.microgrids:
            observations.extend(microgrid.get_state(time_step))

        # Include the current time step in the state
        observations.append(time_step / self.max_time_steps)

        return np.array(observations)

    def _calculate_reward(self):
        total_operational_cost = 0
        total_transaction_cost = 0

        # Loop through each microgrid to calculate the operational cost
        for i, grid in enumerate(self.microgrids):
            # Calculate the operational cost for this microgrid at the current time step
            pv_power_output = grid.solar_panel.get_generation(self.current_time_step, self.max_time_steps)
            operational_cost = grid.calculate_operational_cost(pv_power_output)
            total_operational_cost += operational_cost

            # Get power bought and sold by the microgrid (for simplicity, assuming each grid interacts with main grid)
            P_buy = grid.get_power_from_grid(self.current_time_step)  # Power bought from the grid
            P_sell = grid.get_power_to_grid(self.current_time_step)  # Power sold to the grid

            # Calculate transaction cost with the main grid for this microgrid
            transaction_cost = self.main_grid.calculate_transaction_cost(P_buy, P_sell)
            total_transaction_cost += transaction_cost

        # Total cost is the sum of operational cost and transaction cost
        total_cost = total_operational_cost + total_transaction_cost

        # Reward is the negative of the total cost (minimize cost)
        reward = - total_cost

        # Time penalty (e.g., larger penalty for taking longer)
        time_penalty = -0.1 * self.time_step

        return reward + time_penalty

    def _apply_switch_actions(self, actions):
        # Apply switch control between neighboring microgrids
        for i in range(self.num_microgrids - 1):
            self.microgrids[i].set_switch(actions[i])

    def reset(self, **kwargs):
        self.current_time_step = 0
        return self.state

    def _get_next_state(self):
        state = []
        for grid in self.microgrids:
            state.extend(grid.get_state(0))
        # Include the current time step in the state
        state.append(self.current_time_step / self.max_time_steps)  # Normalized time
        return np.array(state)


