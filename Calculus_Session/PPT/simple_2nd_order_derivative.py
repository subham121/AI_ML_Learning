import numpy as np
import matplotlib.pyplot as plt

# Define the x range
x = np.linspace(-5, 5, 100)

# Define functions
f1 = x ** 2
f2 = -x ** 2
f3 = x ** 3

# First derivatives
f1_1st = 2 * x
f2_1st = -2 * x
f3_1st = 3 * x ** 2

# Second derivatives
f1_2nd = np.full_like(x, 2)
f2_2nd = np.full_like(x, -2)
f3_2nd = 6 * x

# Plotting
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

functions = [(f1, f1_1st, f1_2nd, 'x²'),
             (f2, f2_1st, f2_2nd, '-x²'),
             (f3, f3_1st, f3_2nd, 'x³')]
# functions = [(f1, f1_1st, f1_2nd, 'x²')]
for row, (f, df, d2f, label) in enumerate(functions):
    axes[row][0].plot(x, f, label=f'{label}', color='blue')
    axes[row][0].set_title(f'Function: {label}')
    axes[row][0].axvline(0, color='gray', linestyle='--')
    axes[row][0].grid(True)

    axes[row][1].plot(x, df, label=f"{label}'", color='green')
    axes[row][1].set_title(f'First Derivative: d({label})/dx')
    axes[row][1].axhline(0, color='gray', linestyle='--')
    axes[row][1].grid(True)

    axes[row][2].plot(x, d2f, label=f"{label}''", color='red')
    axes[row][2].set_title(f'Second Derivative: d²({label})/dx²')
    axes[row][2].axhline(0, color='gray', linestyle='--')
    axes[row][2].grid(True)

# Set common labels
for ax in axes[-1]:
    ax.set_xlabel("x")

plt.tight_layout()
plt.show()
