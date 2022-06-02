from tensorflow import keras

from src.common import SAVED_MODELS_BASE_PATH, keras_dataset_image_preprocessing

SAVED_MODELS_MNIST_PATH = SAVED_MODELS_BASE_PATH.joinpath('mnist')

dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = keras_dataset_image_preprocessing(dataset)


def get_mnist_train_test():
    global train_images, test_images
    return train_images, test_images  # scaled pixel values in [0,1]


def get_mnist_labels_categorical():
    global train_labels, test_labels
    return train_labels, test_labels
