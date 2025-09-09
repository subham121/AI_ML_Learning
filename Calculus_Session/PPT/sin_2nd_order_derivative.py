import numpy as np
import matplotlib.pyplot as plt

# Define x values
x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
# print(2 * np.pi)
# Define functions
f = np.sin(x)        # Original function
f_prime = np.cos(x)  # First derivative
f_double_prime = -np.sin(x)  # Second derivative

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot original function
axs[0].plot(x, f, label='f(x) = sin(x)', color='blue', linewidth=2)
axs[0].set_title('Original Function: f(x) = sin(x)')
axs[0].grid(True)
axs[0].legend()

# Plot first derivative
axs[1].plot(x, f_prime, label="f'(x) = cos(x)", color='green', linestyle='--', linewidth=2)
axs[1].set_title("First Derivative: f'(x) = cos(x)")
axs[1].grid(True)
axs[1].legend()

# Plot second derivative
axs[2].plot(x, f_double_prime, label="f''(x) = -sin(x)", color='red', linestyle='-.', linewidth=2)
axs[2].set_title("Second Derivative: f''(x) = -sin(x)")
axs[2].grid(True)
axs[2].legend()

# Axis labels and layout
plt.xlabel("x")
plt.tight_layout()
plt.show()
