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
test = np.array([1, 1, 1, 1])
print(test)
test_1 = test.reshape(1, test.size)
print(test_1)
print(test_1.shape)
print(test.shape)
print(test.ndim)

#%%
print(np.arange(10))
