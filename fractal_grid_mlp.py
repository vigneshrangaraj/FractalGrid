import cvxpy as cp
import numpy as np


class FractalGridOptimizer:
    def __init__(self, num_microgrids=7, M=1000):
        self.num_microgrids = num_microgrids
        self.M = M  # Big-M constant to enforce logical constraints

    def solve_theoretical_optimum_timestep(self, P_load_timestep, P_pv_timestep, rho_timestep):
        """
        Solve the theoretical optimum for a single time step using CVXPY with Big-M method.

        Parameters:
        - P_load_timestep: Load demand for each microgrid at the current time step (1 x N array)
        - P_pv_timestep: PV generation for each microgrid at the current time step (1 x N array)
        - rho_timestep: Locational Marginal Prices (LMP) for the current time step (1 x N array)

        Returns:
        - Optimized values for power bought, sold, generated, stored, and transferred for the current time step.
        """
        N = self.num_microgrids  # Number of microgrids
        beta = 0.1  # Discount factor for selling power

        # Decision Variables (1 x N for each microgrid at the current time step)
        P_buy = cp.Variable(N, nonneg=True)  # Power bought from the grid
        P_sell = cp.Variable(N, nonneg=True)  # Power sold to the grid
        P_DG = cp.Variable(N, nonneg=True)  # Distributed Generation (DG) power; decision of whether I should dispatch or not an dhow much
        P_storage_charge = cp.Variable(N, nonneg=True)  # Power used to charge storage
        P_storage_discharge = cp.Variable(N, nonneg=True)  # Power discharged from storage
        P_transfer = cp.Variable((N, N), nonneg=True)  # Power transferred between microgrids

        # Binary variable to control buying or selling at each microgrid
        buy_or_sell = cp.Variable(N, boolean=True)  # Binary indicator (1 = buy, 0 = sell)

        # Objective: Minimize operational and transaction cost for the current time step
        objective = cp.Minimize(
            cp.sum(cp.multiply(rho_timestep, P_buy) - beta * cp.multiply(rho_timestep, P_sell) + 0.05)  # Grid transaction cost
        )

        # Constraints for each microgrid
        constraints = []
        for i in range(N):
            # Power balance equation: Power in = Power out for each microgrid at the current time step
            power_in = P_buy[i] + P_DG[i] + cp.sum(P_transfer[:, i]) + P_storage_discharge[i]
            power_out = P_load_timestep[i] + P_storage_charge[i] + cp.sum(P_transfer[i, :]) + P_pv_timestep[i] + P_sell[
                i]
            constraints.append(power_in == power_out)

            # Energy storage constraints
            constraints.append(P_storage_charge[i] >= 0)
            constraints.append(P_storage_discharge[i] >= 0)

            # Distributed Generation constraints
            constraints.append(P_DG[i] >= 0)
            constraints.append(P_DG[i] <= 20)

            # constraint on SOC charge and discharge
            constraints.append(P_storage_charge[i] <= 100)
            constraints.append(P_storage_discharge[i] <= 100)

            # Big-M constraints to enforce no simultaneous buying and selling
            constraints.append(P_buy[i] <= self.M * buy_or_sell[i])  # If buying, P_buy can be positive
            constraints.append(P_sell[i] <= self.M * (1 - buy_or_sell[i]))  # If selling, P_sell can be positiv

        # Solve the optimization problem for the current time step
        problem = cp.Problem(objective, constraints)
        optimal_cost = problem.solve()

        return {
            'optimal_cost': optimal_cost,
            'P_buy': P_buy.value,
            'P_sell': P_sell.value,
            'P_DG': P_DG.value,
            'P_storage_charge': P_storage_charge.value,
            'P_storage_discharge': P_storage_discharge.value,
            'P_transfer': P_transfer.value,
            'buy_or_sell': buy_or_sell.value
        }


# Example usage:
if __name__ == "__main__":
    # Example input data for a single time step (random for demonstration purposes)
    num_microgrids = 7
    P_load_timestep = np.random.rand(num_microgrids) * 7  # Load demand for each microgrid at current timestep
    P_pv_timestep = np.random.rand(num_microgrids) * 5  # PV generation for each microgrid at current timestep
    rho_timestep = np.random.rand(num_microgrids) * 5  # Locational Marginal Prices (LMP)

    optimizer = FractalGridOptimizer(num_microgrids=num_microgrids)
    result = optimizer.solve_theoretical_optimum_timestep(P_load_timestep, P_pv_timestep, rho_timestep)

    print("Optimized Cost for Timestep:", result['optimal_cost'])
    print("Power Bought from Grid (P_buy):", result['P_buy'])
    print("Power Sold to Grid (P_sell):", result['P_sell'])
    print("Distributed Generation (P_DG):", result['P_DG'])
    print("Storage Charge (P_storage_charge):", result['P_storage_charge'])
    print("Storage Discharge (P_storage_discharge):", result['P_storage_discharge'])
    print("Power Transfer between Microgrids (P_transfer):", result['P_transfer'])
    print("Buy or Sell Decision (1 = Buy, 0 = Sell):", result['buy_or_sell'])
