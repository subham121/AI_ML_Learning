# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#Step 2: Generate a Tiny Dataset (Linearly Separable)
# Create 2D binary classification dataset (2 features, 2 classes)
X, y = make_classification(n_samples=10, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Add intercept term (bias) by adding a column of ones
X = np.hstack((np.ones((X.shape[0], 1)), X))  # shape: (10, 3)
print(X)
print(y)
# Visualize
plt.scatter(X[y==0][:,1], X[y==0][:,2], color='red', label='Class 0')
plt.scatter(X[y==1][:,1], X[y==1][:,2], color='blue', label='Class 1')
plt.title("Tiny Binary Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
log_h = []
log_1_h = []
# Step 4: Cost Function (Log Loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    log_h.append(np.log(h))
    log_1_h.append(np.log(1-h))
    # cross entropy formula
    return -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
# Step 5: Gradient Descent
def gradient_descent(X, y, theta, alpha, epochs, track_every=100):
    m = len(y)
    cost_history = []
    theta_snapshots = []

    for i in range(epochs):
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        if i % track_every == 0:
            theta_snapshots.append(theta.copy())

    return theta, cost_history, theta_snapshots

#---------------------#
# Track decision boundary evolution
def plot_decision_boundary(theta_vals):
    x1_vals = np.linspace(X[:,1].min(), X[:,1].max(), 100)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0][:,1], X[y==0][:,2], color='red', label='Class 0')
    plt.scatter(X[y==1][:,1], X[y==1][:,2], color='blue', label='Class 1')

    for idx, theta in enumerate(theta_vals):
        x2_vals = -(theta[0] + theta[1]*x1_vals) / theta[2]
        plt.plot(x1_vals, x2_vals, alpha=0.3, label=f'epoch {idx*100}')

    plt.title("📏 Decision Boundary Evolution", fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()
#-----------------------#
# Step 6: Training the Model
# Initialize theta to zeros
# theta = np.zeros(X.shape[1])

# Train using gradient descent
alpha = 0.1      # learning rate
epochs = 1000
theta = np.zeros(X.shape[1])
final_theta, cost_history, theta_snapshots = gradient_descent(X, y, theta, alpha, epochs)
# Plot loss
# plt.plot(range(len(cost_history)), cost_history, color='orange')
# plt.title("Cost vs Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Log Loss")
# plt.grid(True)
# plt.show()

# Plot evolving decision boundaries
plot_decision_boundary(theta_snapshots)


# Step 7: Plot Cost Convergence
# plt.plot(range(epochs), cost_history)
# # plt.plot(range(1), log_h, color='orange')
# # plt.plot(range(1), log_1_h, color='orange')
# plt.title("Cost vs Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Cost")
# plt.grid(True)
# plt.show()

# Step 8: Decision Boundary Visualization
x1_vals = np.linspace(X[:,1].min(), X[:,1].max(), 100)
x2_vals = -(final_theta[0] + final_theta[1]*x1_vals) / final_theta[2]

plt.scatter(X[y==0][:,1], X[y==0][:,2], color='red', label='Class 0')
plt.scatter(X[y==1][:,1], X[y==1][:,2], color='blue', label='Class 1')
plt.plot(x1_vals, x2_vals, color='green', label='Decision Boundary')
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Predict & Accuracy
def predict(X, theta):
    probs = sigmoid(X @ theta)
    return (probs >= 0.5).astype(int)

y_pred = predict(X, final_theta)
accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)
