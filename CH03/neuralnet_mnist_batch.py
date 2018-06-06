#%%
import matplotlib.pyplot as plt
import math
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
    return (x_test, y_test)


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


#%%
img_test, label_test = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0


#%%
for i in range(0, len(img_test), batch_size):
    img_batch = img_test[i:i+batch_size]
    y_batch = predict(network, img_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == label_test[i:i+batch_size])


#%%

print("Accuracy " + str(float(accuracy_cnt)/len(img_test)))


#%%
print(img_test[0:batch_size].shape)
test = predict(network, img_test[0:batch_size])
print(test.shape)
print(test)

#%%
# Testing Area:

print(os.getcwd())
test_x = np.linspace(0, 50, 50, dtype=np.int)
print(test_x)
test_y = np.log(test_x)

plt.plot(test_x, -test_y)
plt.show()
