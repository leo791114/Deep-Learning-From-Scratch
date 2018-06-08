#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
'''
Define a function

'''


def function_2(x, y):
    return x**2+y**2


#%%
'''
Generating data

'''
x = np.linspace(-3, 3, 14)
y = np.linspace(-3, 3, 14)
xv, yv = np.meshgrid(x, y)
z = function_2(xv, yv)

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xv, yv, z, rstride=1, cstride=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x)')
plt.show()
