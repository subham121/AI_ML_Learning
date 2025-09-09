import numpy as np
import matplotlib.pyplot as plt

# -------------Shows the tangent line/slope on a curve--------------#
# Sample loss function: f(x) = (x - 3)^2 + 2
# def f(x): return (x - 3)**2 + 2
# def grad_f(x): return 2*(x - 3)
# x = np.linspace(-6, 6, 1000)
# y = f(x)
# point = -1.0
# # point = 1.0
# slope = grad_f(point)
# tangent = slope * (x - point) + f(point)
#
# plt.plot(x, y, label='f(x)')
# plt.plot(x, tangent, '--', label='Tangent at x=1')
# plt.scatter([point], [f(point)], color='red')
# plt.title("Tangent Line = Derivative")
# plt.legend(); plt.grid(True)
# plt.show()


#----------------------1-D Gradient descent setup--------------------#​
# Sample loss function: f(x) = (x - 3)^2 + 2
def f(x): return (x - 3)**2 + 2
def grad_f(x): return 2*(x - 3)
x_vals = [0]
lr = 0.1
for _ in range(10):
    x_new = x_vals[-1] - lr * grad_f(x_vals[-1])
    x_vals.append(x_new)

# Plot
x = np.linspace(-1, 7, 100)
y = f(x)
plt.plot(x, y, label='Loss Function')
plt.scatter(x_vals, [f(i) for i in x_vals], color='red')
plt.plot(x_vals, [f(i) for i in x_vals], '--r', label='Gradient Descent')
plt.title("Gradient Descent on Loss Function")
plt.xlabel("x"); plt.ylabel("Loss")
plt.legend(); plt.grid(True)
plt.show()