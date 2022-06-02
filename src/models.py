import keras.models
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Reshape, Conv2DTranspose, \
    Input
from keras.models import Sequential, Model

from src.common import get_optimizer, loss_cladec_generator

tf.random.set_seed(1234)


def get_classifier_model_compiled():
    # create VGG like model
    # inspired by https://www.kaggle.com/code/nvsabhilash/keras-vgg-like-architecture-on-mnist-0-98471/notebook
    model = get_classifier_model(add_dense=True)
    model.add(Dense(10, activation='sigmoid'))
    optimizer = get_optimizer()
    model.compile(optimizer, 'categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def get_classifier_model(add_dense: bool):
    # create VGG like model
    # inspired by https://www.kaggle.com/code/nvsabhilash/keras-vgg-like-architecture-on-mnist-0-98471/notebook
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D())
    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D())

    if add_dense:
        classifier.add(Flatten())
        classifier.add(Dropout(0.5))
        classifier.add(Dense(128, activation='relu'))

    return classifier


def get_decoder_out(decoder_in: Input, isDense: bool):
    if isDense:
        decoder = Reshape((1, 1, 128))(decoder_in)
        decoder = Conv2DTranspose(64, (4, 4), padding='valid', strides=(2, 2), activation='relu')(decoder)
        decoder = Conv2DTranspose(32, (3, 3), padding='valid', strides=(2, 2), activation='relu')(decoder)
        decoder = Conv2DTranspose(1, (3, 3), padding='valid', strides=(3, 3), activation='sigmoid', output_padding=1)(
            decoder)
    else:
        decoder = Conv2DTranspose(64, (4, 4), padding='same', strides=(2, 2), activation='relu')(decoder_in)
        decoder = Conv2DTranspose(1, (4, 4), padding='same', strides=(2, 2), activation='sigmoid')(decoder)
    return decoder


def get_rafae(add_dense=True):
    encoder = get_classifier_model(add_dense)
    decoder = get_decoder_out(encoder.output, add_dense)
    return Model([encoder.input], [decoder])


def get_cladec(classifier: keras.models.Model, alpha: float, isDense=True):
    if isDense:
        acts_input = Input(shape=(128,), dtype=tf.float32)  # activations of coded layer
    else:
        acts_input = Input(shape=(7, 7, 64), dtype=tf.float32)
    # custom inputs for the loss function
    img_input = Input(shape=(28, 28))  # training or test images
    labels_input = Input(shape=(10,))  # categorical encoded true labels

    # decoder
    img_output = get_decoder_out(acts_input, isDense=isDense)
    model = Model([acts_input, labels_input, img_input], [img_output])

    model.add_loss(loss_cladec_generator(classifier, alpha)(img_input, labels_input, img_output))
    return model
