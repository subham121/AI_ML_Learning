import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(0)
m = 100
X = np.random.randn(m, 2)           # two features
true_theta = np.array([2, -1])     # underlying weights
bias = -0.5
z = X @ true_theta + bias
probs = 1 / (1 + np.exp(-z))
y = (probs >= 0.5).astype(int)      # labels 0 or 1

# 2. Prepare for training
X_bias = np.hstack([np.ones((m,1)), X])  # add bias term
theta = np.zeros(3)                      # initialize weights [b, w1, w2]
α = 0.1                                  # learning rate
iters = 100
loss_hist = []

# 3. Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 4. Gradient descent loop
for i in range(iters):
    z = X_bias @ theta
    h = sigmoid(z)
    # cross-entropy loss
    loss = -np.mean(y*np.log(h) + (1-y)*np.log(1-h))
    loss_hist.append(loss)
    # gradient
    grad = (X_bias.T @ (h - y)) / m
    theta -= α * grad

# 5. Plot decision boundary
plt.figure()
for c in [0,1]:
    plt.scatter(X[y==c,0], X[y==c,1], label=f'Class {c}')
x_vals = np.array(plt.gca().get_xlim())
# boundary: θ0 + θ1 x + θ2 y = 0  ⇒ y = -(θ0 + θ1 x)/θ2
y_vals = -(theta[0] + theta[1]*x_vals) / theta[2]
plt.plot(x_vals, y_vals, label='Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Decision Boundary')

# 6. Plot loss curve
plt.figure()
plt.plot(loss_hist)
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Loss')
plt.title('Loss over Iterations')

plt.show()
