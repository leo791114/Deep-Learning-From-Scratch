#%%
import os
import sys
print(sys.path)
from Ch05.layer_naive import MulLayer, AddLayer

#%%
'''
Layers defining
'''
apple_num = 2
apple_price = 100
orange_num = 3
orange_price = 150
tax = 1.1

apple_mul_layer = MulLayer()
orange_mul_layer = MulLayer()
tax_mul_layer = MulLayer()
apple_orange_add_layer = AddLayer()

#%%
'''
Forward
'''
apple_total_price = apple_mul_layer.forward(apple_num, apple_price)  # 1
orange_total_price = orange_mul_layer.forward(orange_num, orange_price)  # 2
apple_orange_price = apple_orange_add_layer.forward(
    apple_total_price, orange_total_price)  # 3
total_price = tax_mul_layer.forward(apple_orange_price, tax)  # 4


#%%
'''
Backward
'''
dout = 1
d_apple_orange, d_tax = tax_mul_layer.backward(dout)  # 4
d_apple_total_price, d_orange_total_price = apple_orange_add_layer.backward(
    d_apple_orange)  # 3
d_apple_num, d_apple_price = apple_mul_layer.backward(d_apple_total_price)  # 2
d_orange_num, d_orange_price = orange_mul_layer.backward(
    d_orange_total_price)  # 1

#%%
'''
Results
'''
print(d_apple_orange, d_tax)
print(d_apple_total_price, d_orange_total_price)
print(int(d_apple_num), d_apple_price)
print(int(d_orange_num), d_orange_price)
