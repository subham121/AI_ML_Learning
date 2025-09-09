import numpy as np
import matplotlib.pyplot as plt

# Probability values from just above 0 to just below 1
p = np.linspace(1e-5, 1 - 1e-5, 1000)

# Log functions
log_p = np.log(p)
log_1_minus_p = np.log(1 - p)

# Cross-entropy loss
loss_y1 = -np.log(p)         # when y = 1
loss_y0 = -np.log(1 - p)     # when y = 0

# Plotting
plt.figure(figsize=(10, 6))

# Basic log terms
# plt.plot(p, log_p, label='log(p)', linestyle='--', color='blue')
# plt.plot(p, log_1_minus_p, label='log(1 - p)', linestyle='--', color='red')

# Cross-entropy loss
plt.plot(p, loss_y1, label='Log Loss when y = 1', color='blue')
plt.plot(p, loss_y0, label='Log Loss when y = 0', color='red')

plt.axhline(0, color='black', linestyle='--')
plt.title("📉 Log(p), Log(1-p), and Log Loss Curves")
plt.xlabel("p (Predicted Probability)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.ylim(-10, 10)  # Focused range
plt.show()
