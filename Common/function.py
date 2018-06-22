
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

#%%


def cross_entropy_error(y_output, y_label):
      # If there is only one data, change the output from row vector to column vector
      #ex: [[1,2,3,4]]
    if y_output.ndim == 1:
        y_output = y_output.reshape(1, y_output.size)
        y_label = y_label.reshape(1, y_label.size)
    # Dealing one-hot-label:
    if y_output.size == y_label.size:
        y_label = y_label.argmax(axis=1)

    batch_size = y_output.shape[0]
    # Small delta is used to avoid situation like np.log(0) which will return negative infinite.
    # y_output[np.arange(batch_size), y_label] can deal with label representation
    return -np.sum(np.log(y_output[np.arange(batch_size), y_label] + 1e-7)) / batch_size
