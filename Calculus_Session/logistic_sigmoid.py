import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1. Generate classification data
X, y = make_classification(n_samples=10, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# 2. Add intercept (bias) term
X = np.hstack((np.ones((X.shape[0], 1)), X))  # shape: (10, 3)

# 3. Initialize random theta (weights)
theta = np.array([0.5, 1.0, -1.0])  # sample weights

# 4. Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 5. Compute z = X @ theta and sigmoid(z)
z = X @ theta
sigmoid_vals = sigmoid(z)

# 6. Plot sigmoid values vs z
plt.figure()
plt.plot(z, sigmoid_vals, 'bo', label='Sigmoid outputs')
plt.xlabel('z = θᵀx')
plt.ylabel('Sigmoid(z)')
plt.title('Sigmoid Activation for Data Points')
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
plt.grid(True)
plt.legend()
plt.show()
