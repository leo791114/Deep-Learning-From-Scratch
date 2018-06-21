#%%
import os
import sys
sys.path.append(os.pardir)
import numpy as np
from Dataset.load_mnist import load_mnist
from PIL import Image
import pickle
from Common.function import sigmoid, softmax
np.set_printoptions(threshold=np.nan)

#%%
'''
Define error function - Mean Squared Error
'''


def mean_squared_error(y, t):
    if y.ndim == 1:
        return 0.5 * np.sum((y-t)**2)
    else:
        batch_size = y.shape[0]
        print(batch_size)
        return (1/batch_size)*np.sum((y-t)**2, axis=1)


#%%
'''
Run Mean Squared Error Function
'''
test_y_mean = np.array([[1 for x in range(0, 10)] for y in range(0, 11)])
test_t_mean = np.array([[1 if x == y else 0 for x in range(0, 10)]
                        for y in range(0, 11)])
test_mean = mean_squared_error(test_y_mean, test_t_mean)
print(test_mean)

#%%
'''
Define Cross Entropy Error Function - One Hot Label
'''


def cross_entropy_function(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    delta = 1e-7  # Use to prevent infinite number
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + delta)) / (batch_size)


#%%
'''
Define Cross Entropy Error Function - Label
'''


def cross_entropy_function_label(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


#%%
'''
Run Cross Entropy Error Function - One Hot Label
'''
test_y_cross = np.array([[x for x in range(0, 10)] for y in range(0, 11)])
test_t_cross = np.array(
    [[1 if x == y else 0 for x in range(0, 10)] for y in range(0, 11)])
test_cross = cross_entropy_function(test_y_cross, test_t_cross)
print(test_cross)


#%%
