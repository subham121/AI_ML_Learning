import numpy as np
import matplotlib.pyplot as plt

# Loss function and its derivative
theta = np.linspace(-1, 7, 100)
J = (theta )**2 - 2*theta
dJ = 2 * theta - 2

plt.figure(figsize=(12, 6))

# Plot Loss Function
plt.subplot(1, 2, 1)
plt.plot(theta, J, label='Function J(θ)')
plt.axvline(3, color='gray', linestyle='--', label='Minimum')
plt.title("Loss Function J(θ)")
plt.xlabel("θ")
plt.ylabel("J(θ)")
plt.legend()
plt.grid(True)

# Plot Derivative
plt.subplot(1, 2, 2)
plt.plot(theta, dJ, color='orange', label="Gradient dJ/dθ")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(3, color='gray', linestyle='--', label='Zero Gradient')
plt.title("Gradient of J(θ)")
plt.xlabel("θ")
plt.ylabel("dJ/dθ")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
