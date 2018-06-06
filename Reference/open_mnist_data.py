#%%
try:
    import urllib.request
except ImportError:
    raise ImportError("You should use Python 3.x")
import numpy as np
import matplotlib.pyplot as plt
import os.path
import gzip
import pickle
np.set_printoptions(threshold=np.nan)

#%%
img_size = 784

#%%
print(os.getcwd())
print(os.path.abspath(os.getcwd()))

#%%
# Download Mnist
url_base = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
url_base_label = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

#%%
# it has to be a file name
file_path = os.path.abspath(os.getcwd()) + '/' + 'train-images-idx3-ubyte.gz'
file_path_label = os.path.abspath(
    os.getcwd()) + '/' + 'train-labels-idx1-ubyte.gz'


def _download(file, url_path):
    if os.path.exists(file):
        print(file + " already exist")
        return
    urllib.request.urlretrieve(url_path, file)


_download(file_path, url_base)
_download(file_path_label, url_base_label)

# urllib.request.urlretrieve(url_base, file_path)
# urllib.request.urlretrieve(url_base_label, file_path_label)

#%%
# Open zip file 'file_path'
with gzip.open(file_path, 'rb') as f1:
    img_data = np.frombuffer(f1.read(),  dtype='uint8', offset=16)
    img_data = img_data.reshape(-1, img_size)
    print(img_data.shape)
    print('Done')

#%%
print(file_path_label)
with gzip.open(file_path_label, 'rb') as f2:
    label_data = np.frombuffer(f2.read(), dtype='uint8', offset=8)
    print(label_data.shape)
    print("Done")


# %%
# Create pickle file
def _create_pickle(file_name):
    save_file = os.getcwd() + '/' + 'mnist_test.pkl'
    print(save_file)
    print("Creating pickle file...")

    with open(save_file, 'wb') as f:
        pickle.dump(file_name, f, -1)

    print("Done!")

#%%
# Create one hot label


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


#%%
plt.matshow(np.reshape(img_data[0:1, ], (28, 28)), cmap=plt.get_cmap('gray'))

#%%
# Save pickle file
_create_pickle(img_data)


#%%
print(img_data[0:1, ].astype(np.float32)/255.0)
print(img_data[0:1, ].size)
print(img_data.size)
#rows: 60000 ; columns: 784
print(img_data.shape)
print(img_data.dtype)

#%%
# flatten data
test_img = img_data
test_img = test_img.reshape(-1, 1, 28, 28)
print(img_data.shape)
print(test_img.shape)
print(test_img[:2, ])

#%%
print(label_data.shape)
print(label_data.size)
print(label_data)

#%%
# One_hot_label
one_hot_label_data = _change_one_hot_label(label_data)
print(one_hot_label_data)
