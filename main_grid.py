class MainGrid:
    def __init__(self, Pmax_g, LMP_buy, LMP_sell_discount, delta_t=1):
        self.Pmax_g = Pmax_g
        self.LMP_buy = LMP_buy
        self.LMP_sell = LMP_buy * LMP_sell_discount  # Discounted price for selling power
        self.delta_t = delta_t  # Duration of each time step (hours)

    def calculate_transaction_cost(self, P_buy, P_sell):
        if P_buy > 0 and P_sell > 0:
            if P_buy >= P_sell:
                P_sell = 0  # Prioritize buying power
            else:
                P_buy = 0  # Prioritize selling power

        P_buy = min(P_buy, self.Pmax_g)
        P_sell = min(P_sell, self.Pmax_g)

        # Transaction cost: Buying incurs cost, selling earns revenue
        transaction_cost = (P_buy * self.LMP_buy - P_sell * self.LMP_sell) * self.delta_t

        return transaction_cost