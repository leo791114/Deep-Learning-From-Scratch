#%%
import os
import sys
sys.path.append(os.pardir)
import numpy as np
from Common.function import softmax, cross_entropy_error
from Common.gradient import numerical_gradient


#%%
'''
Define the net
'''


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # intialize the weight to 2*3 array

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, y_label):
        z = self.predict(x)  # z would modify according to weight change
        y_predict = softmax(z)
        loss = cross_entropy_error(y_predict, y_label)
        return loss


#%%
'''
Run the simpleNet
'''
x = np.array([0.6, 0.9])
y_label = np.array([0, 0, 1])

net = simpleNet()
print(net.W)

y_predict = net.predict(x)
print(y_predict)

print(net.loss(x, y_label))

#%%
'''
Run Gradient Descent
'''


def f(w): return net.loss(x, y_label)


dw = numerical_gradient(f, net.W)
print(dw)
