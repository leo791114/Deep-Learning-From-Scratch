#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# Activation Function - Step Function


def step_function(x):
    y = x > 0
    return y.astype(np.int)


x_step = np.linspace(-5.0, 5.0, 500)
print(type(x_step))
y_step = step_function(x_step)
plt.plot(x_step, y_step)
plt.ylim(-0.1, 1.1)
plt.show()

#%%
# Activation Function - Sigmoid Function


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


x_sigmoid = np.arange(-5, 5, 0.02)
print(type(x_sigmoid))
y_sigmoid = sigmoid_function(x_sigmoid)
plt.plot(x_sigmoid, y_sigmoid)
plt.ylim(-0.1, 1.1)
plt.show()


#%%
# Activation Function - ReLU (Rectified )

def ReLU_function(x):
    return np.maximum(0, x)


x_relu = np.linspace(-5, 5, 500)
y_relu = ReLU_function(x_relu)
plt.plot(x_relu, y_relu)
plt.ylim(-1, 5)
plt.show()

#%%
# Activation Function - Softmax (Probability)


def Softmax_Function(x):
    c = np.max(x)
    exp_a = np.exp(x - c)  # Prevention for overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


test = np.array([0.1, 0.3, 0.5])
test_y = Softmax_Function(test)
print(test_y, np.sum(test_y))

#%%
plt.plot(x_step, y_step, label="Step_Function")
plt.plot(x_sigmoid, y_sigmoid, label="Sigmoid_Function")
plt.plot(x_relu, y_relu, label="ReLU_Function")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()

#%%
# Neural Network Multiplication - Forward Propagation
# Layer 0 - Activation Function: Sigmoid Function 
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.4, 0.6], [0.2, 0.4, 0.5]])
B1 = np.array([0.1, 0.2, 0.3])
print(X.shape, W1.shape, B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid_function(A1)
print(A1, Z1)

# Layer 1 - Activation Function: Sigmoid Function
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape, W2.shape, B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid_function(A2)
print(A2, Z2)

# Layer 2 (output layer) - Activation Function: Identity Function (For Regression)


def identity_function(x):
    return x


W3 = np.array([[0.1, 0.2], [0.3, 0.4]])
B3 = np.array([0.3, 0.4])
print(Z2.shape, W3.shape, B3.shape)

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(A3, Y)

#%%
# Neural Network Multiplication (Integration)


def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.4, 0.6], [0.2, 0.4, 0.5]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.2], [0.3, 0.4]])
    network["b3"] = np.array([0.3, 0.4])

    return network


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
