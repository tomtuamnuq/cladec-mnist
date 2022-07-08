import os
import pathlib

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
import src.utils
from src.cladec import ClaDec, RefAE
from src.classifier import create_classifier_model_compiled
from src.utils import keras_dataset_image_preprocessing, get_optimizer, DENSE_LAYER_NAME, CONV_LAYER_NAME, ALPHAS

tf.random.set_seed(1234)
file_path = os.path.dirname(os.path.realpath("__file__"))
home_dir = pathlib.Path(file_path).parent
os.chdir(home_dir)

EPOCHS = 5


def train_classifier(x, y):
    classifier = create_classifier_model_compiled()
    classifier.fit(x, y, epochs=EPOCHS * 2)
    return classifier


def train_cladec(x, y, classifier: Sequential, layer_name: str, alpha: int, decoder: Model = None):
    claDec = ClaDec(classifier, layer_name, alpha / 100, decoder)
    claDec.compile(optimizer=get_optimizer(learning_rate=0.01))
    claDec.fit(x, y, epochs=EPOCHS)
    return claDec


def train_refae(x, cladec: Model):
    refAE = RefAE(cladec)
    refAE.compile(optimizer=get_optimizer(learning_rate=0.01))
    refAE.fit(x, epochs=EPOCHS)
    return refAE


def create_models():
    datasets = keras.datasets.fashion_mnist, keras.datasets.mnist
    model_paths = src.utils.SAVED_MODELS_PATH_FASHION, src.utils.SAVED_MODELS_PATH_MNIST
    layers = [DENSE_LAYER_NAME, CONV_LAYER_NAME]
    for dataset, saved_models_path in zip(datasets, model_paths):
        (train_images, train_labels_c), _ = keras_dataset_image_preprocessing(dataset)
        classifier = train_classifier(train_images, train_labels_c)
        classifier.save(saved_models_path.joinpath('classifier'))
        print("Saved classifier to ", saved_models_path.joinpath('classifier'))
        for alpha in ALPHAS:
            saved_models_path_alpha = saved_models_path.joinpath('cladec').joinpath(f'{alpha * 100:2.0f}')
            saved_models_path_refae = saved_models_path.joinpath('refae')
            for layer_name in layers:
                decoder = None
                if layer_name == 'my_dense':
                    decoder = ClaDec.create_128_dense_decoder()
                elif layer_name == 'my_conv':
                    decoder = ClaDec.create_64_conv_decoder()
                cladec = train_cladec(train_images, train_labels_c, classifier, layer_name, alpha, decoder)
                cladec.save(saved_models_path_alpha.joinpath(layer_name))
                print("Saved cladec to ", saved_models_path_alpha.joinpath(layer_name))
                if alpha == 0:
                    refae = train_refae(train_images, cladec)
                    refae.save(saved_models_path_refae.joinpath(layer_name))
                    print("Saved refae to ", saved_models_path_refae.joinpath(layer_name))


if __name__ == '__main__':
    create_models()
