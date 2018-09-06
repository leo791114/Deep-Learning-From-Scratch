#%%
import numpy as np

#%%
'''
Numerical Gradient Function for One-Dimensional Data
'''


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
    return grad


#%%
'''
Define Numerical Gradient Function for Two-Dimensional Data
'''


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)

    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


#%%
'''
Define a General Numerical Gradient Function
'''


def numerical_gradient(f, x):
    h = 1e-4  # 0.001
    grad = np.zeros_like(x)
    # print(grad)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        # print(fxh1)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)  # f(x-h)
        # print(fxh2)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # print(grad)

        x[idx] = tmp_val  # return x to original value
        it.iternext()

    return grad
