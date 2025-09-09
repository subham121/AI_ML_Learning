import numpy as np
import matplotlib.pyplot as plt

# Define input range
x = np.linspace(-10, 10, 400)

# Define 5 functions
y1 = 5*x + 10                     # Linear: y = 5x + 10
y2 = x**2                     # Quadratic: y = x^2
y3 = np.sin(x)                # Sine curve: y = sin(x)
y4 = np.exp(-x**2)            # Gaussian: y = exp(-x^2)
y5 = 2*(x**3) + (5*x) + 4     # third degree polynomial: 2x^3 + 5x + 4

# Create the plot
plt.figure(figsize=(10, 8))
# fig, axes = plt.subplots(5, 1, figsize=(15, 10), sharex=True)

# plt.plot(x, y1, label='y = 5x + 10 (Linear)')
# plt.plot(x, y2, label='y = x² (Quadratic)')
# plt.plot(x, y3, label='y = sin(x) (Sine)')
# plt.plot(x, y4, label='y = exp(-x²) (Gaussian)')
plt.plot(x, y5, label='y = 2x^3 + 5x + 4 (cubic polynomial function)')

# Plot formatting
plt.title("Various Mathematical Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='gray', lw=1)   # x-axis
plt.axvline(0, color='gray', lw=1)   # y-axis
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Show for different values of x how y changes
# and for minor change in the parameter how the graph deflects
