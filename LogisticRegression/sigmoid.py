import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
# create data
x = np.linspace(-10, 10, 30)
print(x)

# get sigmoid output
y = sigmoid(x)
rounded_y = np.round(y, 3)
print(rounded_y)

# get derivative of sigmoid
d = d_sigmoid(x)