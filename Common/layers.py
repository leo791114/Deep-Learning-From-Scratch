#%%
import numpy as np
from Common.function import *

#%%
'''
Activation Function Layer: ReLU (Rectified Linear Unit)
'''


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # if x <= 0, return true
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


#%%
'''
Activation Function Layer: Sigmoid
'''


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)  # out = 1 / (1 - np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):

        dx = dout * self.out * (1 - self.out)

        return dx


#%%
test = np.array([[1, 2], [3, 4], [5, 6]])
print(test.shape)
print(test)
print(test[2][0])
