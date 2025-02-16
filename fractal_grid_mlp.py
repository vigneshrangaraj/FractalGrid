import numpy as np
import cvxpy as cp
from sympy.printing.numpy import const

from microgrid import MicroGrid
from utils import config


class FractalGridOptimizer:
    def __init__(self, num_microgrids=3, M=1000, seed=42):
        np.random.seed(seed)  # Fix random seed for reproducibility
        self.num_microgrids = num_microgrids
        self.M = M  # Big-M constant for logical constraints
        self.config = config

        # Initialize microgrids with a fractal tree structure
        self.microgrids, self.total_switches = self.initialize_microgrids(num_microgrids, config.MAX_MG_POWER)

    def initialize_microgrids(self, num_microgrids, max_power):
        """Initialize microgrids using a fractal tree structure."""
        microgrids = []
        switch_set = set()

        for i in range(num_microgrids):
            microgrid = MicroGrid(1, max_power)
            microgrids.append(microgrid)

        for i in range(num_microgrids):
            left_child = 2 * i + 1  # Left child index
            right_child = 2 * i + 2  # Right child index

            if left_child < num_microgrids:
                switch_name = f"S_{min(i, left_child)}_to_{max(i, left_child)}"
                microgrids[i].add_neighbor(microgrids[left_child], switch_name)
                microgrids[left_child].add_neighbor(microgrids[i], switch_name)
                switch_set.add(switch_name)

            if right_child < num_microgrids:
                switch_name = f"S_{min(i, right_child)}_to_{max(i, right_child)}"
                microgrids[i].add_neighbor(microgrids[right_child], switch_name)
                microgrids[right_child].add_neighbor(microgrids[i], switch_name)
                switch_set.add(switch_name)

        total_switches = len(switch_set)
        return microgrids, total_switches

    def step(self, timesteps):
        """Perform optimization for multiple timesteps with transitions."""
        cumulative_cost = 0
        results = []

        for t in range(timesteps):
            # Collect current states from microgrids
            states = [grid.get_state(t) for grid in self.microgrids]

            # Extract PV, SOC, and load from states
            P_pv = np.array([state[0] for state in states])
            SOC = np.array([state[1] for state in states])
            P_load = np.array([state[2] for state in states])
            local_net_powers = np.array([state[3] for state in states])

            # Decision Variables
            P_buy = cp.Variable(self.num_microgrids, nonneg=True)
            P_sell = cp.Variable(self.num_microgrids, nonneg=True)
            P_DG = cp.Variable(self.num_microgrids, nonneg=True)
            P_storage_charge = cp.Variable(self.num_microgrids, nonneg=True)
            P_storage_discharge = cp.Variable(self.num_microgrids, nonneg=True)

            # Binary variables for actions
            z_charge = cp.Variable(self.num_microgrids, boolean=True)
            z_discharge = cp.Variable(self.num_microgrids, boolean=True)
            z_buy = cp.Variable(self.num_microgrids, boolean=True)
            z_sell = cp.Variable(self.num_microgrids, boolean=True)

            maintenance_cost = config.PV_MAINTANENCE_COST

            for grid in self.microgrids:
                grid.degradation_factor -= config.PV_DEGRADATION_RATE


            # Objective: Minimize operational cost
            objective = cp.Minimize(
                cp.sum(
                    P_buy * 20
                ) +
                cp.sum(
                    maintenance_cost + (1 - config.PV_DEGRADATION_FACTOR) * P_DG +
                    (1 - config.PV_INVERTER_EFFICIENCY) * P_DG
                )
                - cp.sum(P_sell * 20 * 0.1)
            )

            # Constraints
            constraints = []

            # Add Constraints
            for i in range(self.num_microgrids):
                soc_i = self.microgrids[i].ess.soc

                # charge constraints
                available_capacity = (self.microgrids[i].ess.soc_max - self.microgrids[i].ess.soc) * self.microgrids[i].ess.capacity  # Max energy that can be added
                max_charging_power = min(available_capacity,
                                         config.ESS_MAX_CHARGE_POWER)  # Ensure we don't exceed SOC limit
                actual_charging_power = min(local_net_powers[i], max_charging_power)
                if actual_charging_power > 0:
                    constraints.append(P_storage_charge[i] <= actual_charging_power)
                else:
                    constraints.append(P_storage_charge[i] == 0)

                # discharge constraints
                available_capacity = (self.microgrids[i].ess.soc - self.microgrids[i].ess.soc_min) * self.microgrids[i].ess.capacity  # Max energy that can be removed
                max_discharging_power = min(available_capacity,
                                            config.ESS_MAX_DISCHARGE_POWER)  # Ensure we don't exceed SOC limit
                actual_discharging_power = min(local_net_powers[i], max_discharging_power)
                if actual_discharging_power > 0:
                    constraints.append(P_storage_discharge[i] <= actual_discharging_power)
                else:
                    constraints.append(P_storage_discharge[i] == 0)

                # constraints
                max_discharging_power = soc_i * self.microgrids[i].ess.capacity
                constraints.append(P_storage_discharge[i] <= max_discharging_power)
                constraints.append(P_DG[i] + P_buy[i] - P_sell[i] + P_storage_discharge[i] == P_load[i] + P_storage_charge[i] )
                constraints.append(P_DG[i] <= self.microgrids[i].solar_panel.get_generation(t, timesteps))
                constraints.append(P_storage_charge[i] <= max(0,self.config.ESS_MAX_CHARGE_POWER))
                constraints.append(P_storage_discharge[i] <= max(0, self.config.ESS_MAX_DISCHARGE_POWER))
                constraints.append(P_storage_charge[i] >= 0)
                constraints.append(P_storage_discharge[i] >= 0)
                constraints.append(P_DG[i] >= 0)
                constraints.append(P_DG[i] <= self.config.MAX_MG_POWER)

            # Solve the Optimization Problem
            problem = cp.Problem(objective, constraints)
            cost = problem.solve(solver=cp.GUROBI, verbose=False)

            local_net_powers = []
            processed_switches = set()

            # discharge and charge
            for i in range(self.num_microgrids):
                P_discharge_values = P_storage_discharge.value
                P_charge_values = P_storage_charge.value

                if P_discharge_values[i] > 0:
                    self.microgrids[i].ess.discharge(P_discharge_values[i], 1)
                if P_charge_values[i] > 0:
                    self.microgrids[i].ess.charge(P_charge_values[i], 1, t)

            for i, grid in enumerate(self.microgrids):
                local_net_power = grid.calculate_local_net_power(P_pv[i], P_load[i])
                local_net_powers.append(local_net_power)

            # Neighbor Power Exchanges
            for i, grid in enumerate(self.microgrids):
                for neighbor in grid.neighbors:
                    neighbor_index = neighbor.index
                    switch_name = f"S_{min(i, neighbor_index)}_to_{max(i, neighbor_index)}"

                    if switch_name not in processed_switches:
                        local_net_power_i = local_net_powers[i].value if isinstance(local_net_powers[i],
                                                                                    cp.Expression) else \
                        local_net_powers[i]
                        local_net_power_neighbor = local_net_powers[neighbor_index].value if isinstance(
                            local_net_powers[neighbor_index], cp.Expression) else local_net_powers[neighbor_index]

                        if local_net_power_i >= 1e-6 and local_net_power_neighbor <= -1e-6:
                            power_transferred = min(local_net_power_i, abs(local_net_power_neighbor))
                            local_net_powers[i] -= power_transferred
                            local_net_powers[neighbor_index] += power_transferred
                        elif local_net_power_i <= -1e-6 and local_net_power_neighbor >= 1e-6:
                            power_transferred = min(local_net_power_neighbor, abs(local_net_power_i))
                            local_net_powers[i] += power_transferred
                            local_net_powers[neighbor_index] -= power_transferred

                        processed_switches.add(switch_name)
                grid.final_net_power = local_net_powers[i]

            # Log Results
            cumulative_cost += cost
            results.append({
                'time': t,
                'cost': cumulative_cost,
                'P_buy': P_buy.value,
                'P_sell': P_sell.value,
                'P_DG': P_DG.value,
                'P_storage_charge': P_storage_charge.value,
                'P_storage_discharge': P_storage_discharge.value,
                'final_net_power': local_net_powers
            })
        return results, cumulative_cost


# Example Usage
if __name__ == "__main__":
    optimizer = FractalGridOptimizer(num_microgrids=7)
    results, total_cost = optimizer.step(timesteps=24)
    print("Total Cost:", total_cost)
    for res in results:
        print(res)