import sys
import os
import pathlib

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import src.utils
from src.cladec import ClaDec, RefAE
from src.classifier import train_classifier
from src.utils import (keras_dataset_image_preprocessing,
                       LAYERS_TO_EXPLAIN,
                       DATASETS,
                       MODEL_PATHS,
                       ALPHAS,
                       SAVED_EVAL_BASE_PATH, )

tf.random.set_seed(1234)
file_path = os.path.dirname(os.path.realpath("__file__"))
home_dir = pathlib.Path(file_path).parent
os.chdir(home_dir)


def eval_created_models(epochs: int):
    columns = ["dataset", "layer", "eval", "classifier", "refAE"] + [f'{alpha:2}' for alpha in
                                                                     ALPHAS]
    df = pd.DataFrame(columns=columns)
    for dataset, saved_models_path in zip(DATASETS, MODEL_PATHS):
        (train_images, train_labels_c), (
            test_images, test_labels_c) = keras_dataset_image_preprocessing(dataset)
        classifier = keras.models.load_model(saved_models_path.joinpath('classifier'))
        print("Loaded classifier from ", saved_models_path.joinpath('classifier'))
        classifier_loss, classifier_acc = classifier.evaluate(test_images, test_labels_c)
        saved_models_path_refae = saved_models_path.joinpath('refae')
        for layer_name in LAYERS_TO_EXPLAIN:
            decoder = None
            if layer_name == src.utils.DENSE_LAYER_NAME:
                decoder = ClaDec.create_128_dense_decoder()
            elif layer_name == src.utils.CONV_LAYER_NAME:
                decoder = ClaDec.create_64_conv_decoder()
            total_losses = [classifier_loss]
            reconstruction_losses = [None]
            class_losses = [classifier_loss]
            class_accuracies = [classifier_acc]
            for alpha in ALPHAS:
                saved_models_path_alpha = saved_models_path.joinpath('cladec').joinpath(f'{alpha:2}').joinpath(
                    layer_name)
                cladec = ClaDec.create_from_weights(saved_models_path_alpha.joinpath(layer_name),
                                                    classifier,
                                                    layer_name,
                                                    alpha / 100,
                                                    decoder)
                print("Evaluating ", saved_models_path_alpha)
                rec_loss, class_loss, total_loss = cladec.evaluate(test_images, test_labels_c)
                print("Training Evaluation Classifier for ", saved_models_path_alpha)
                cladec_eval = train_classifier(cladec.predict(train_images), train_labels_c, epochs)
                _, cladec_eval_acc = cladec_eval.evaluate(cladec.predict(test_images),
                                                          test_labels_c)
                if alpha == 0:
                    saved_models_path_refae_layer = saved_models_path_refae.joinpath(layer_name)
                    refae = RefAE.create_from_weights(saved_models_path_refae_layer.joinpath(
                        layer_name), cladec)
                    print("Evaluating ", saved_models_path_refae)
                    ref_loss = refae.evaluate(test_images)
                    print("Training Evaluation Classifier for ", saved_models_path_refae_layer)
                    refae_eval = train_classifier(refae.predict(train_images),
                                                  train_labels_c,
                                                  epochs)
                    _, refae_eval_acc = refae_eval.evaluate(refae.predict(test_images),
                                                            test_labels_c)
                    total_losses.append(ref_loss)
                    reconstruction_losses.append(ref_loss)
                    class_losses.append(None)
                    class_accuracies.append(refae_eval_acc)

                total_losses.append(total_loss)
                reconstruction_losses.append(rec_loss)
                class_losses.append(class_loss)
                class_accuracies.append(cladec_eval_acc)
            df.loc[len(df)] = [saved_models_path.name, layer_name, "total_loss", *total_losses]
            df.loc[len(df)] = [saved_models_path.name, layer_name, "reconstruction_loss",
                               *reconstruction_losses]
            df.loc[len(df)] = [saved_models_path.name, layer_name, "classification_loss",
                               *class_losses]
            df.loc[len(df)] = [saved_models_path.name, layer_name, "evaluation_accuracy",
                               *class_accuracies]

    df.to_csv(SAVED_EVAL_BASE_PATH.joinpath("evaluation_result.csv"))


if __name__ == '__main__':
    sys.stdout = open(SAVED_EVAL_BASE_PATH.joinpath("std_output.txt"), "w")
    eval_created_models(10)
