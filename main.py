# Import relevant packages
import math
import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(123456)

# importing dataframe
# Had to convert xlsx to csv in order for my environment to read the dataframe
df = pd.read_csv('CashflowModel.csv')

'''
Geometric Brownian Motion Monte Carlo Simulation
'''


class Simulation:

    def __init__(self, Path_Count, Step_Count):
        self.Path_Count = Path_Count
        self.Step_Count = Step_Count

    def GBM(self, T, mu, sigma, S0):

        # Setting up grid for the stock price, time and the brownian motion term
        W = np.random.normal(0.0, 1.0, [Path_Count, Step_Count])
        S = np.zeros([Path_Count, Step_Count + 1])
        time = np.zeros([Step_Count + 1])

        S[:, 0] = (S0)  # Establish the first term as the starting price
        dt = T / float(Step_Count)  # define dt
        for i in range(0, Step_Count):
            # This if statement is done to normalize the values simulated from the random normal distribution.
            # It will be impossible have W as normal distribution unless normalized
            if Path_Count > 1:
                W[:, i] = (W[:, i] - np.mean(W[:, i])) / np.std(W[:, i])

            # Model GBM using stock prices
            S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * np.power(sigma, 2)) * dt + sigma * W[:, i] * np.sqrt(dt))
            # Record time for plotting graphs
            time[i + 1] = time[i] + dt
        # Save dictionary for ease of recall
        paths = {"time": time, "S": S}
        return paths


# Define parameters
Step_Count = 5
Path_Count = 1000
T = 5
r = 0.00
sigma = 0.093
S0 = 1.37625
K = S0
Notional = 100000000

Sims = Simulation(Path_Count=Path_Count, Step_Count=Step_Count)
Paths = Sims.GBM(T, r, sigma, S0)
time = Paths["time"]
S = Paths["S"]


def plot_GBM():
    plt.figure("GBM simulation")
    plt.plot(time, np.transpose(S))
    plt.grid()
    plt.xlabel("Time (years)")
    plt.ylabel("S(t)")
    plt.title("GBM Simulation Path")
    plt.show()


plot_GBM()

'''
Calculating Internal Rate of Return
'''

# Take the mean of each time step as the monte carlo price of GBP/USD for that year
S_av = np.mean(S, axis=0)
# Since there is no drift term, the number floats around the original price point on 1.37625

# Save the cash flows in their own list for ease of calculation
# Convert cf from string to float
cf = [float(df.iloc[i, 3].replace(',', '')) for i in range(len(df))]

# Multiply the cash flows of each year with their corresponding FX rate for each simulation
CF_USD = cf * S

# Calculate the IRR of each simulations cash flows
# numpy has its own irr function built in
irr = [npf.irr(CF_USD[i, :]) for i in range(len(CF_USD))]


# plot distribution
def distribution_plot():
    plt.figure("Distribution of IRR")
    plt.hist(irr, bins=20)
    plt.title("Distribution of IRR in portfolio USD currency")
    plt.xlabel("IRR")
    plt.grid()
    plt.show()


distribution_plot()

# Evaluate the percentiles
fifth = np.percentile(irr, 5)
fifty = np.percentile(irr, 50)
ninety_five = np.percentile(irr, 95)
print(f"5th percentile IRR:    {fifth: .4f}")
print(f"50th percentile IRR:   {fifty: .4f}")
print(f"95th percentile IRR:   {ninety_five: .4f}")

'''
Using Puts to hedge against currency risk
Calculating Put premium using Monte Carlo results
'''
# Hedging
# Calculating the options fair price
S_final = S[:, -1]  # Only interested in the 2026 expiry data
# Take the strike price minus the stock price of each simulation, then take the maximum value that or 0
# Then take the average price of the put prices as the MC put price
# Since r=0, no need to worry about discount factor

# Calculate payoff of each simulation
MC_Put_Payoff = [np.max([0, K - S_final[i]]) for i in range(len(S_final))]
# Calculate MC premium
MC_Put_prem = np.mean(MC_Put_Payoff)  # Put Premium
print(f"Monte Carlo Put Premium: {MC_Put_prem: .4f}")
print(f"Notional spending on Put: ${MC_Put_prem * Notional: .4f}")

'''
Calculating Internal Rate of Return Including in the Hedged Puts Values
'''

# Build an array to add the hedged cash flows to
cf_hedged = CF_USD
# Add premium to trade date cash flow
cf_hedged[:, 0] = cf_hedged[:, 0] - MC_Put_prem * Notional
# Add payoff to the the final value at expiry date
cf_hedged[:, -1] = cf_hedged[:, -1] + np.where(S_final < K, K - S_final, 0) * Notional

# Calculate the irr with the cash flows
irr_new = [npf.irr(cf_hedged[i, :]) for i in range(len(cf_hedged))]

distibutions = ['Unhedged', 'Hedged']
def distribution_plot_hedged():
    plt.figure("Distribution of IRR with Put")
    plt.hist(irr_new, bins=20, color='red')
    plt.title("IRR Distribution of USD Cash Flows with Put Currency Hedge")
    plt.xlabel("IRR")
    plt.grid()
    plt.show()


distribution_plot_hedged()

distibutions = ['Unhedged', 'Hedged']
def distribution_comparison():
    plt.figure("Distribution comparison")
    plt.hist(irr, bins=20, alpha=0.5)
    plt.hist(irr_new, bins=20, color='red', alpha=0.5)
    plt.title("IRR Distribution Comparison Between \n Hedged and Unhedged Cashflows")
    plt.xlabel("IRR")
    plt.legend(distibutions)
    plt.grid()
    plt.show()


distribution_comparison()

# Evaluate the percentiles
fifth_hedged = np.percentile(irr_new, 5)
fifty_hedged = np.percentile(irr_new, 50)
ninety_five_hedged = np.percentile(irr_new, 95)
print(f"5th percentile Hedged IRR:    {fifth_hedged: .4f}")
print(f"50th percentile Hedged IRR:   {fifty_hedged: .4f}")
print(f"95th percentile Hedged IRR:   {ninety_five_hedged: .4f}")



