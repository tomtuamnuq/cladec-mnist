import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(1234)
dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)
train_images /= 255.  # maximum pixel value
test_images /= 255.

SAVED_MODELS_BASE_PATH = pathlib.Path(__file__).parent.parent.joinpath('saved-models')


def get_mnist_train_test():
    global train_images, test_images
    return train_images, test_images  # scaled pixel values in [0,1]


def get_mnist_labels_categorical():
    global train_labels, test_labels
    train_labels_c = keras.utils.to_categorical(train_labels)
    test_labels_c = keras.utils.to_categorical(test_labels)
    return train_labels_c, test_labels_c


def remove_data_in_last_dim(data: np.ndarray, split_percentage: int):
    # remove 100-split_percentage of the data in the last dimension (-1)
    subset_nr = int(data.shape[-1] * split_percentage / 100)
    return data[:, ..., :subset_nr]


def get_optimizer(learning_rate=0.001):
    return keras.optimizers.Adam(
        learning_rate=learning_rate,  # default 0.001
        epsilon=1e-04,  # default 1e-07
    )


def save_layer_activations(path: pathlib.Path, model: keras.models.Model, layer_name: str, test_data=test_images,
                           train_data=train_images,
                           train_split: int = 10):
    layer = keras.Model(inputs=model.input,
                        outputs=model.get_layer(layer_name).output)
    test_acts = layer(test_data)
    np.save(path.joinpath('test'), test_acts.numpy())
    # training data in batches due to memory requirements of GPU
    for i, data in enumerate(np.array_split(train_data, train_split)):
        train_acts = layer(data)
        np.save(path.joinpath(f'train_{i}'), train_acts.numpy())


def load_layer_activations(classifier_path: pathlib.Path, train_split: int = 10):
    dense_path = classifier_path.joinpath('dense')
    dense_test_acts = np.load(dense_path.joinpath('test.npy'))
    dense_train_acts = np.concatenate([np.load(dense_path.joinpath(f'train_{i}.npy')) for i in range(train_split)])
    conv_path = classifier_path.joinpath('conv')
    conv_test_acts = np.load(conv_path.joinpath('test.npy'))
    conv_train_acts = np.concatenate([np.load(conv_path.joinpath(f'train_{i}.npy')) for i in range(train_split)])
    return (dense_train_acts, dense_test_acts), (conv_train_acts, conv_test_acts)


def loss_cladec_generator(classifier: keras.models.Model, alpha: float):
    mse = keras.losses.MeanSquaredError()
    cce = keras.losses.CategoricalCrossentropy()

    def loss_cladec(img_input, labels_input, img_output):
        return (1 - alpha) * mse(img_output, img_input) + alpha * cce(classifier(img_output), labels_input)

    return loss_cladec
