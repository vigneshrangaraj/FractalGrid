import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate x values
x = np.linspace(-5, 5, 100)

# Calculate the corresponding y values for the normal distribution
mean = 0
std_dev = 1
y = norm.pdf(x, mean, std_dev)

# Plot the curve
plt.plot(x, y)
plt.title("Bell-shaped Curve (Normal Distribution)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.show()