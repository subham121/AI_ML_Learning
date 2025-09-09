# Import libraries
import numpy as np

# Step 1: Define Activation Function and its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Step 2: Define Loss Function - Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 3: Define Input and Output Data
# 4 training samples with 3 features each
X = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

# Corresponding target output for each sample
y = np.array([[0], [1], [1], [0]])

# Step 4: Initialize Parameters
np.random.seed(42)
input_size = 3
hidden_size = 2
output_size = 1
learning_rate = 0.1

# Random weights and zero biases
W1 = np.random.randn(input_size, hidden_size)  # weights between input and hidden (3x2)
W2 = np.random.randn(hidden_size, output_size) # weights between hidden and output (2x1)
b1 = np.zeros((1, hidden_size))
b2 = np.zeros((1, output_size))

# Step 5: Training Loop with Backpropagation
for epoch in range(10000):
    # ----- Forward Pass -----
    Z1 = np.dot(X, W1) + b1     # Input to hidden layer
    A1 = sigmoid(Z1)            # Hidden layer activation
    Z2 = np.dot(A1, W2) + b2    # Hidden to output layer
    A2 = sigmoid(Z2)            # Output prediction

    # ----- Loss -----
    loss = mse(y, A2)

    # ----- Backward Pass -----
    # Output layer error
    dA2 = A2 - y                             # Gradient of loss
    dZ2 = dA2 * sigmoid_derivative(Z2)       # Gradient after activation (chain rule)

    # Hidden layer error
    dA1 = np.dot(dZ2, W2.T)                  # Propagate error back
    dZ1 = dA1 * sigmoid_derivative(Z1)       # Again apply chain rule

    # ----- Update Weights -----
    W2 -= learning_rate * np.dot(A1.T, dZ2)  # Weight update (hidden→output)
    b2 -= learning_rate * np.sum(dZ2, axis=0, keepdims=True)

    W1 -= learning_rate * np.dot(X.T, dZ1)   # Weight update (input→hidden)
    b1 -= learning_rate * np.sum(dZ1, axis=0, keepdims=True)

    # ----- Print Loss -----
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final Predictions
print("\nFinal predictions after training:")
print(A2.round(3))
