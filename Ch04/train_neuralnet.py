#%%
import os
import sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from Dataset.load_mnist import load_mnist
from Ch04.two_layer_net import TwoLayerNet
np.set_printoptions(threshold=np.nan)
print(os.getcwd())

#%%
'''
Define the neural net
'''
# Load Data

(x_train, y_train), (x_test, y_test) = load_mnist(
    normalize=True, flatten=True, one_hot_label=True)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# Hyperparameter
iters_num = 10000  # Setting number of iterations according to our needs
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

#%%

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    # print(batch_mask)
    x_batch = x_train[batch_mask]
    # print(x_batch.shape)
    y_batch = y_train[batch_mask]
    # print(y_batch.shape)

    # Calculate Gradient
    # grad = network.numerical_gradient(x_batch, y_batch)
    grad = network.gradient(x_batch, y_batch)

    # Update weights
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Calculate losses
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        print(i)
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc |" + str(train_acc) + ',' + str(test_acc))

#%%
'''
Draw  Graphs of Accuracy v.s Epochs and Loss v.s Iterations
'''
fig, axes = plt.subplots(nrows=2, ncols=1)
print(axes)

markers = {'train': 'o', 'test': 's'}

x = np.arange(len(train_acc_list))
x2 = np.arange(iters_num)
# Accuracy v.s. Epochs
axes[0].plot(x, train_acc_list, label='train accuracy',
             marker=markers['train'])
axes[0].plot(x, test_acc_list, label='test accuracy',
             marker=markers['test'], linestyle='--')
axes[0].set_xlabel('epochs')
axes[0].set_ylabel('accuracy')
axes[0].set_ylim(0, 1.0)
axes[0].legend(loc='lower right')

# Loss v.s. Iterations
axes[1].plot(x2, train_loss_list, label='train loss')
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='lower right')
plt.tight_layout()
plt.show()

'''
Only plot one graph
'''
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train accuracy', marker=markers['train'])
# plt.plot(x, test_acc_list, label='test accuracy',
#          marker=markers['test'], linestyle='--')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()
