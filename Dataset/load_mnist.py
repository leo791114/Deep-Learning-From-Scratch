'''
 Reference:
1. https://ithelp.ithome.com.tw/articles/10186473
'''

#%%
try:
    import urllib.request
except ImportError:
    raise ImportError("You should use Python 3.x")
import os.path
import numpy as np
import gzip
import pickle


#%%
url_base = "http://yann.lecun.com/exdb/mnist/"
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + '/mnist.pkl'

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

#%%
# Initialization


def init_minst():
    downlaod_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file!")
    with open(save_file, 'wb') as f:
        #-1 (a negative number) will select the highest protocol
        pickle.dump(dataset, f, -1)
    print("Done!")


#%%
# Call _download function to download mnist data


def downlaod_mnist():
    for mnist_data in key_file.values():
        _download(mnist_data)


#%%
# Download specific mnist data from the internet using urllib.request.urlretrieve


def _download(file_name):
    file_path = dataset_dir + '/' + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + "...")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done.")


#%%
# Load image


def _load_img(file_name):
    file_path = dataset_dir + '/' + file_name

    print("Converting " + file_name + " to Numpy array")
    with gzip.open(file_path, 'rb') as f:
        img_data = np.frombuffer(f.read(), np.uint8, offset=16)
    img_data = img_data.reshape(-1, img_size)
    print(img_data.shape)
    print("Done")

    return img_data


#%%
# Load label

def _load_label(file_name):
    file_path = dataset_dir + '/' + file_name

    print("Converting " + file_name + " to Numpy array")
    with gzip.open(file_path, 'rb') as f:
        label_data = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done!")

    return label_data

#%%
# Processing mnist data


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


#%%
# Create one_hot_label
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

#%%
# Load mnist data


def load_mnist(normalize=True, flatten=True, one_hot_label=False):

    if not os.path.exists(save_file):
        init_minst()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_labe'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return ((dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']))


#%%
# If the Python script is run by itself, then call the init_mnist() function
if __name__ == '__main__':
    init_minst()


#%%
print(os.path)
print(os.getcwd())
print(os.path.dirname(__file__))
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))
