#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#%%
def numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # return to the original value

    return grad


#%%
'''
Define numerical gradient function with batch
'''


def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_no_batch(f, x)

        return grad


#%%
'''
Define Function
'''


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


#%%
'''
Define Tangent Line Function
'''


def tangent_line(f, x):
    m = numerical_gradient(f, x)
    print(m)
    delta_y = f(x) - m*x
    return lambda t: m*t + delta_y


#%%
'''
Draw the gradient graph of x^2 + y^2
'''
if __name__ == '__main__':  # tell the script is run by itself or imported as a module
    print(__name__)

    x = np.arange(-2, 2.5, 0.25)
    y = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x, y)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    # , headwidth=10, scale=40, color='#444444')
    plt.quiver(X, Y, grad[0], grad[1], angles='xy', color='#666666')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


#%%
test = np.arange(-2, 2)
print(test.ndim)
test_diff = numerical_gradient_no_batch(function_2, test)
print(test_diff)
