
#%%
import numpy as np

#%%
# Identity Function


def identity_function(x):
    return x

#%%
# Step Function


def step_function(x):
    return np.array(x > 0, dtype=np.int)

#%%
# Sigmoid function


def sigmoid(x):
    return 1/(1+np.exp(-x))

#%%
# ReLU Function


def relu(x):
    return np.maximum(0, x)

#%%
# ReLU Grad Function


def relu_grad(x):
    grad = np.zeros(x)
    grad[x > 0] = 1
    return grad

#%%
# Softmax


def softmax(x):
    # Why?
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x-np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))
    return y
