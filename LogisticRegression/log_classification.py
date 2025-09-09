import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data
dib = pd.read_csv('Diabetes+Example+Data.csv')
dib['Diabetes'] = dib['Diabetes'].map({'Yes': 1, 'No': 0})
X = dib['Blood Sugar Level'].values
y = dib['Diabetes'].values

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Negative log-likelihood
def neg_log_likelihood(beta):
    z = beta[0] + beta[1] * X
    p = sigmoid(z)
    # Avoid log(0)
    p = np.clip(p, 1e-8, 1-1e-8)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

# Minimize negative log-likelihood
result = minimize(neg_log_likelihood, [0, 0])
beta0, beta1 = result.x

# Plot negative log-likelihood surface
beta0_range = np.linspace(beta0-2, beta0+2, 100)
beta1_range = np.linspace(beta1-2, beta1+2, 100)
B0, B1 = np.meshgrid(beta0_range, beta1_range)
Z = np.array([[neg_log_likelihood([b0, b1]) for b0, b1 in zip(row_b0, row_b1)] for row_b0, row_b1 in zip(B0, B1)])

plt.figure(figsize=(8,6))
plt.contourf(B0, B1, Z, levels=50, cmap='viridis')
plt.scatter(beta0, beta1, color='red', label='Best Fit')
plt.xlabel('Intercept (beta0)')
plt.ylabel('Slope (beta1)')
plt.title('Negative Log-Likelihood Surface')
plt.legend()
plt.colorbar(label='Neg Log-Likelihood')
plt.show()

# Calculate predicted probabilities and odds
z = beta0 + beta1 * X
p = sigmoid(z)
odds = p / (1 - p)

# Plot odds
plt.figure(figsize=(8,6))
plt.scatter(X, odds, c=y, cmap='bwr', label='Odds')
plt.xlabel('Blood Sugar Level')
plt.ylabel('Odds')
plt.title('Odds vs Blood Sugar Level')
plt.colorbar(label='Diabetes')
plt.show()

# Plot best fit curve
plt.figure(figsize=(8,6))
plt.scatter(X, y, label='Data', alpha=0.5)
X_curve = np.linspace(X.min(), X.max(), 100)
p_curve = sigmoid(beta0 + beta1 * X_curve)
plt.plot(X_curve, p_curve, color='red', label='Best Fit Curve')
plt.xlabel('Blood Sugar Level')
plt.ylabel('Probability of Diabetes')
plt.title('Logistic Regression Fit')
plt.legend()
plt.show()