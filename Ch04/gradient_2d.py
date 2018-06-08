#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
test = np.array([1, 2])
for idx in range(test.size):
    tmp = test[idx]
    test[idx] = tmp + 1
    print(test)
    print(test[idx])
    test[idx] = tmp - 1
    print(test)
    print(test[idx])
    test[idx] = tmp
    print(test)
    print(test[idx])
