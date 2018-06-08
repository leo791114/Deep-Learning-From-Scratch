#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
'''
Numerical differentiation：
Central differentiation
'''


def numerical_diff(f, x):
    '''
    Here, we can't use 1e-50 due to rounding error.
    Python will interpret np.float32(1e-50) as 0
    '''
    h = 1e-4  # 0.0001
    return (f(x+h)-f(x-h))/(2*h)


#%%
'''
Define a testing function
'''


def function_1(x):
    return 0.01*x**2 + 0.1*x


#%%
'''
Define tangent line of function_1(x)：
Consider the tangent line at P(a,b).
f'(a) = ma + (f(a) - ma) = f(a) = b.
'''


def tangent_line(f, x):
    s = numerical_diff(f, x)
    print(s)
    delta_y = f(x) - s*x
    return lambda t: t*s + delta_y



#%%
'''
Drawing function_1 from 0 to 20 by 0.1 steps
'''
x = np.arange(0.0, 20.0, 0.1)

y = function_1(x)

tangent_f = tangent_line(function_1, 5)
y2 = tangent_f(x)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.plot(x, y2)
plt.plot(5, function_1(5), 'ro')
plt.show()
