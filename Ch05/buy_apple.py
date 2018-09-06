#%%
import os
import sys
from Ch05.layer_naive import MulLayer, AddLayer


#%%
apple = 100
apple_amout = 2
tax = 1.1

#%%
# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#%%
# forward
apple_price = mul_apple_layer.forward(apple_amout, apple)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

#%%
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple_num, dapple = mul_apple_layer.backward(dapple_price)

print(dapple_num, dapple, dtax)
