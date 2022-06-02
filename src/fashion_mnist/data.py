from tensorflow import keras

from src.common import keras_dataset_image_preprocessing

dataset = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = keras_dataset_image_preprocessing(dataset)


def get_fashion_mnist_train_test():
    global train_images, test_images
    return train_images, test_images  # scaled pixel values in [0,1]


def get_fashion_mnist_labels_categorical():
    global train_labels, test_labels
    return train_labels, test_labels
