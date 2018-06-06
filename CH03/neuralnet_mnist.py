
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
def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    return x_test, y_test

#%%


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

#%%


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sofmax(a3)

    return y


#%%
img_test, label_test = get_data()
network = init_network()
print(img_test.shape, label_test.shape)
print(img_test.shape)
print(img_test[0].shape)
print(network['W1'].shape)
print(network['W2'].shape)
print(network['W3'].shape)

#%%
accuracy_count = 0
for i in range(len(img_test)):
    y = predict(network, img_test[i])
    p = np.argmax(y)
    if p == label_test[i]:
        accuracy_count += 1

print("Accuracy:" + str(float(accuracy_count)/len(img_test)))
