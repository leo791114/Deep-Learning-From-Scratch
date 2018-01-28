#%%
import numpy as np

# #%%
# test_1 = np.array([1,2,3])
# test_2 = np.array([2,3,4])
# test = np.sum([test_1, test_2], axis=1)
# print(test)

#%%
# Perception - AND Gate


def AND(x1_and, x2_and):
    x_and = np.array([x1_and, x2_and])  # w is weight, theta is threshold
    w_and = np.array([0.5, 0.5])
    b_and = -0.7  # b is bias
    temp_and = np.sum(x_and * w_and) + b_and
    if temp_and <= 0:
        return 0
    elif temp_and > 0:
        return 1


print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))  # truth table

#%%
# Perception - NAND Gate


def NAND(x1_nand, x2_nand):
    x_nand = np.array([x1_nand, x2_nand])
    w_nand = np.array([-0.5, -0.5])
    b_nand = 0.7
    temp_nand = np.sum(x_nand*w_nand) + b_nand
    if temp_nand <= 0:
        return 0
    elif temp_nand > 0:
        return 1


print(NAND(0, 0), NAND(1, 0), NAND(0, 1), NAND(1, 1))

#%%
# Perception - OR Gate


def OR(x1_or, x2_or):
    x_or = np.array([x1_or, x2_or])
    w_or = np.array([0.5, 0.5])
    b_or = 0
    temp_or = np.sum(x_or*w_or) + b_or
    if temp_or <= 0:
        return 0
    elif temp_or > 0:
        return 1


print(OR(0, 0), OR(1, 0), OR(0, 1), OR(1, 1))
