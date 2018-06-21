#%%
import os
import sys
# append parent directory into current working path
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Ch04.gradient_2d import numerical_gradient

#%%
'''
Define Gradient Descent Function
'''


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        # print(x_history)

        grad = numerical_gradient(f, x)
        x -= lr*grad
        # print(x)

    return x, np.array(x_history)


#%%
'''
Define a Function
'''


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


#%%
'''
Run Gradient Descent Method
'''
init_x = np.array([-3.0, 4.0])
result, result_history = gradient_descent(
    function_2, init_x=init_x, lr=0.1, step_num=20)
# print(result)
print(result_history)

fig = plt.figure()
ax = fig.gca()
ax.set_aspect('equal')

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
for i in range(5):
    circle = plt.Circle((0, 0), radius=i, fill=False,
                        color='r', linestyle='--')
    ax.add_artist(circle)
# circle1 = plt.Circle((0, 0), radius=3, fill=False, color='r', linestyle='--')
plt.plot(result_history[:, 0], result_history[:, 1], 'o')

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.xticks(np.arange(-4, 5))
plt.yticks(np.arange(-4, 5))
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()
