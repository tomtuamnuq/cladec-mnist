import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(1234)
dataset = keras.datasets.mnist
train, test = dataset.load_data()
(train_images, train_labels) = train
(test_images, test_labels) = test


def get_mnist_train_test():
    global train_images, test_images
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    train_images /= 255.  # maximum pixel value
    test_images /= 255.
    return train_images, test_images  # scaled pixel values in [0,1]


def get_mnist_labels_categorical():
    global train_labels, test_labels
    train_labels_c = keras.utils.to_categorical(train_labels)
    test_labels_c = keras.utils.to_categorical(test_labels)
    return train_labels_c, test_labels_c


def remove_last_dim(data: np.ndarray, split_percentage: int):
    # remove 100-split_percentage of the data in the last dimension (-1)
    subset_nr = int(data.shape[-1] * split_percentage / 100)
    return data[:, ..., :subset_nr]


def get_optimizer(learning_rate=0.001):
    return keras.optimizers.Adam(
        learning_rate=learning_rate,  # default 0.001
        epsilon=1e-04,  # default 1e-07
    )
