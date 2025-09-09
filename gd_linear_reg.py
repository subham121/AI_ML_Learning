import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🎮 Descent Quest – Gradient Descent Game")

mode = st.selectbox("Mode", ["Linear Regression", "Logistic Regression"])
lr = st.slider("Learning Rate", 0.001, 1.0, 0.1)
iterations = st.slider("Iterations", 10, 500, 50)

# Generate sample data
X = np.linspace(-5, 5, 100)
if mode == "Linear Regression":
    y = 2 * X + 1 + np.random.randn(*X.shape) * 2
    cost_fn = lambda m, b: np.mean((m*X + b - y)**2)
elif mode == "Logistic Regression":
    y = (X > 0).astype(int)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    cost_fn = lambda m, b: -np.mean(y*np.log(sigmoid(m*X+b)) + (1-y)*np.log(1-sigmoid(m*X+b)))

# Gradient descent
m, b = np.random.randn(), np.random.randn()
path = []
for _ in range(iterations):
    if mode == "Linear Regression":
        dm = -2*np.mean(X*(y - (m*X+b)))
        db = -2*np.mean(y - (m*X+b))
    else:
        pred = sigmoid(m*X+b)
        dm = np.mean((pred - y) * X)
        db = np.mean(pred - y)
    m -= lr * dm
    b -= lr * db
    path.append(cost_fn(m, b))

# Plot cost over iterations
fig, ax = plt.subplots()
ax.plot(range(iterations), path, marker='o')
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.set_title(f"Cost Reduction – {mode}")
st.pyplot(fig)

st.write(f"Final m: {m:.2f}, Final b: {b:.2f}")
