#%%
import os
import sys
sys.path.append(os.pardir)
import numpy as np
from Common.function import *
from Common.gradient import numerical_gradient

#%%
'''
Define the Net:
1. input_size: define the amount of neurons in input layer.
2. hidden_size: define the amount of neurons in hidden layer.
3. output_size: define the amount of neurons in output layer.
'''


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
        # Initialization
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, y_label):
        y_predict = self.predict(x)

        return cross_entropy_error(y_predict, y_label)

    def accuracy(self, x, y_label):
        y_predict = self.predict(x)
        y_predict = np.argmax(y_predict, axis=1)
        y_label = np.argmax(y_label, axis=1)

        accuracy = np.sum(y_predict == y_label) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, y_label):
        def loss_W(W): return self.loss(x, y_label)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, y_label):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y-y_label) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


#%%
'''
Run TwoLayerNet
'''
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)

# #%%
# x = np.random.rand(10, 784)  # simulate 10 input data with size of 784
# y = net.predict(x)
# print(x)
# print(y)


# #%%
# y_label = np.random.rand(10, 10)  # simulate 10 label data with size of 10

# grads = net.numerical_gradient(x, y_label)
# print(grads['W1'].shape)
# print(grads['b1'].shape)
# print(grads['W2'].shape)
# print(grads['b2'].shape)
