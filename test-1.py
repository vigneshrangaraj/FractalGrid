import matplotlib.pyplot as plt
import numpy as np

time_steps = np.arange(0, 24, 1)  # 24 time steps (one per hour)
switch_states = np.random.randint(0, 2, (5, len(time_steps)))  # Random binary states for 5 switches

plt.figure(figsize=(10, 6))
for i, switch in enumerate(switch_states):
    plt.step(time_steps, switch, label=f'Switch {i+1}', where='mid')
plt.xlabel('Time (hours)')
plt.ylabel('Switch State (0: Open, 1: Closed)')
plt.title('Switching States Over 24 Hours')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(switch_states, cmap='Blues', cbar=True, xticklabels=time_steps, yticklabels=[f'Switch {i+1}' for i in range(5)])
plt.xlabel('Time (hours)')
plt.ylabel('Switch')
plt.title('Switching States Heatmap Over 24 Hours')
plt.show()

energy_transfer = np.random.rand(5, len(time_steps)) * 10  # Random energy transfers (5 microgrid pairs)

plt.figure(figsize=(10, 6))
plt.stackplot(time_steps, energy_transfer, labels=[f'Grid {i+1} to Grid {i+2}' for i in range(5)])
plt.xlabel('Time (hours)')
plt.ylabel('Energy Transferred (kWh)')
plt.title('Energy Transferred Between Microgrids Over 24 Hours')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

inflow = np.random.rand(5, len(time_steps)) * 10  # Random inflow
outflow = np.random.rand(5, len(time_steps)) * 10  # Random outflow

bar_width = 0.35
index = np.arange(5)

plt.figure(figsize=(10, 6))
for i in range(len(time_steps)):
    plt.bar(index + i*bar_width, inflow[:, i], bar_width, label=f'Inflow at t={i}')
    plt.bar(index + i*bar_width, -outflow[:, i], bar_width, label=f'Outflow at t={i}', alpha=0.7)

plt.xlabel('Microgrids')
plt.ylabel('Energy (kWh)')
plt.title('Energy Inflow/Outflow for Microgrids Over 24 Hours')
plt.xticks(index + bar_width / 2, [f'MG {i+1}' for i in range(5)])
plt.legend()
plt.grid(True)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Switching States
for i, switch in enumerate(switch_states):
    ax1.step(time_steps, switch, label=f'Switch {i+1}', where='mid')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Switch State')
ax1.set_title('Switching States Over 24 Hours')
ax1.grid(True)
ax1.legend()

# Plot 2: Energy Transferred
ax2.stackplot(time_steps, energy_transfer, labels=[f'Grid {i+1} to Grid {i+2}' for i in range(5)])
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Energy Transferred (kWh)')
ax2.set_title('Energy Transferred Between Microgrids Over 24 Hours')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.show()
