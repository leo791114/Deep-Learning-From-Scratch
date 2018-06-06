'''
PIL refer to Python Image Library
'''
#%%
import os
import sys
sys.path.append(os.pardir)
import numpy as np
from Dataset.load_mnist import load_mnist
from PIL import Image

# Define a function to show the image
#%%


def img_show(input_img):
    '''
    Same as below
    pil_img = Image.fromarray(input_img)
    '''
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# Load image data
#%%
(x_train,  y_train), (x_test,  y_test) = load_mnist(
    normalize=False, flatten=True, one_hot_label=False)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Assign the first image
#%%

img = x_train[0]
label = y_train[0]
# print(img)
print(label)
print(img.shape)

img = img.reshape(28, 28)
print(img.shape)

#%%

img_show(img)
