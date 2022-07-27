import sys
import os
import pathlib

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
import src.utils
from src.cladec import ClaDec, RefAE
from src.classifier import train_classifier
from src.utils import (keras_dataset_image_preprocessing,
                       get_optimizer,
                       LAYERS_TO_EXPLAIN,
                       DATASETS,
                       MODEL_PATHS, )

tf.random.set_seed(1234)
file_path = os.path.dirname(os.path.realpath("__file__"))
home_dir = pathlib.Path(file_path).parent
os.chdir(home_dir)

alphas = [99.9, 99.99, 100]  # [0, 1, 5, 10, 25, 50, 99]


def train_cladec(x, y, classifier: Sequential, layer_name: str, alpha: float, epochs: int,
                 decoder: Model = None):
    claDec = ClaDec(classifier, layer_name, alpha / 100, decoder)
    claDec.compile(optimizer=get_optimizer(learning_rate=0.005))
    claDec.fit(x, y, epochs=epochs)
    return claDec


def train_refae(x, cladec: Model, epochs: int):
    refAE = RefAE(cladec)
    refAE.compile(optimizer=get_optimizer(learning_rate=0.005))
    refAE.fit(x, epochs=epochs)
    return refAE


def create_models(epochs: int, load_classifier: True):

    for dataset, saved_models_path in zip(DATASETS, MODEL_PATHS):
        (train_images, train_labels_c), _ = keras_dataset_image_preprocessing(dataset)
        if load_classifier:
            classifier = keras.models.load_model(saved_models_path.joinpath('classifier'))
            print("Loaded classifier from ", saved_models_path.joinpath('classifier'))
        else:
            classifier = train_classifier(train_images, train_labels_c, epochs)
            classifier.save(saved_models_path.joinpath('classifier'))
            print("Saved classifier to ", saved_models_path.joinpath('classifier'))
        saved_models_path_refae = saved_models_path.joinpath('refae')
        for alpha in alphas:
            saved_models_path_alpha = saved_models_path.joinpath('cladec').joinpath(f'{alpha:2}')
            for layer_name in LAYERS_TO_EXPLAIN:
                decoder = None
                if layer_name == src.utils.DENSE_LAYER_NAME:
                    decoder = ClaDec.create_128_dense_decoder()
                elif layer_name == src.utils.CONV_LAYER_NAME:
                    decoder = ClaDec.create_64_conv_decoder()
                cladec = train_cladec(train_images,
                                      train_labels_c,
                                      classifier,
                                      layer_name,
                                      alpha,
                                      epochs,
                                      decoder)
                cladec.save_weights(saved_models_path_alpha.joinpath(layer_name).joinpath(layer_name))
                print("Saved cladec to ", saved_models_path_alpha.joinpath(layer_name))
                if alpha == 0:
                    refae = train_refae(train_images, cladec, epochs)
                    refae.save_weights(saved_models_path_refae.joinpath(layer_name).joinpath(
                        layer_name))
                    print("Saved refae to ", saved_models_path_refae.joinpath(layer_name))


if __name__ == '__main__':
    sys.stdout = open(src.utils.SAVED_MODELS_BASE_PATH.joinpath("std_output_more_alphas.txt"), "w")
    create_models(10, True)  # 10  # test_create_models()
