import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(1234)


def keras_dataset_image_preprocessing(dataset):
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    train_images /= 255.  # maximum pixel value
    test_images /= 255.
    train_labels_c = keras.utils.to_categorical(train_labels)
    test_labels_c = keras.utils.to_categorical(test_labels)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    return (train_images, train_labels_c), (test_images, test_labels_c)


def get_optimizer(learning_rate=0.001):
    return keras.optimizers.Adam(
        learning_rate=learning_rate,  # default 0.001
        epsilon=1e-04,  # default 1e-07
    )


SAVED_MODELS_BASE_PATH = pathlib.Path(__file__).parent.parent.joinpath('saved-models')
SAVED_PICS_PATH = pathlib.Path(__file__).parent.parent.joinpath('eval').joinpath('img')

SAVED_MODELS_PATH_FASHION = SAVED_MODELS_BASE_PATH.joinpath('fashion_mnist')
SAVED_CLASSIFIER_PATH_FASHION = SAVED_MODELS_PATH_FASHION.joinpath('classifier')
SAVED_REFAE_PATH_FASHION = SAVED_MODELS_PATH_FASHION.joinpath('refae')
SAVED_CLADEC_PATH_FASHION = SAVED_MODELS_PATH_FASHION.joinpath('cladec')

SAVED_MODELS_PATH_MNIST = SAVED_MODELS_BASE_PATH.joinpath('mnist')
SAVED_CLASSIFIER_PATH_MNIST = SAVED_MODELS_PATH_MNIST.joinpath('classifier')
SAVED_REFAE_PATH_MNIST = SAVED_MODELS_PATH_MNIST.joinpath('refae')
SAVED_CLADEC_PATH_MNIST = SAVED_MODELS_PATH_MNIST.joinpath('cladec')

DENSE_LAYER_NAME = 'my_dense'
CONV_LAYER_NAME = 'my_conv'
ALPHAS = [0, 1, 50, 99]
