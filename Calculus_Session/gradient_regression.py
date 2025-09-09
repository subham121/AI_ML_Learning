import numpy as np
import matplotlib.pyplot as plt

# 1. Create synthetic data
# np.random.seed(42)
# x = np.linspace(0, 100, 100)
# y = x + np.random.randn(100) * 20
x = np.array([0, 1, 2, 3, 4, 5, 6, 7.])
y = np.array([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])
# 2. Initialize parameters
w = np.random.randn()
b = np.random.randn()
lr = 0.01 #1e-4
epochs = 1000
losses = []

# 3. Gradient descent loop
for epoch in range(epochs):
    y_pred = w * x + b
    error = y_pred - y
    loss = np.mean(error**2)
    losses.append(loss)
    # Gradients
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)
    # Update
    w -= lr * dw
    b -= lr * db
    # if epoch % 50 == 0:
    print(f"(50)Epoch {epoch}, Loss = {loss:.5f}, W = {w:.5f}, B = {b:.5f}, dW = {dw:.5f}, dB = {db:.5f}")
    # if (epoch+1) % 250 == 0:
    # print(f"Epoch {epoch+1:4d}/{epochs}, loss = {loss:.4f}")

print(f"\nFinal line: y ≈ {w:.3f}·x + {b:.3f}")

# 4. Plot results
plt.figure(figsize=(8,6))
plt.scatter(x, y, label='data')
plt.plot(x, w * x + b, color='red', label=f"Fit: y={w:.2f}x+{b:.2f}")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Linear Regression (NumPy manual GD)")
plt.show()

# 5. Loss curve
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.show()