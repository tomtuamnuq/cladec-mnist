import keras.models
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Reshape, Conv2DTranspose, \
    Input
from keras.models import Sequential, Model

from src.common import get_optimizer, loss_cladec_generator

tf.random.set_seed(1234)


def get_classifier_model_compiled():
    classifier = get_classifier_model()
    optimizer = get_optimizer()
    classifier.compile(optimizer, 'categorical_crossentropy', metrics=['categorical_accuracy'])

    return classifier


def get_classifier_model(add_dense=True):
    # create VGG like model
    # inspired by https://www.kaggle.com/code/nvsabhilash/keras-vgg-like-architecture-on-mnist-0-98471/notebook
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu', strides=2))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D())

    classifier.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu', strides=2))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D())
    classifier.add(Flatten())

    if add_dense:
        classifier.add(Dropout(0.5))
        classifier.add(Dense(10, activation='softmax'))

    return classifier


def get_decoder_out(decoder_in: Input):
    decoder = Dense(49, activation='relu')(decoder_in)  # 49 = 7 x 7
    decoder = Reshape((7, 7, 1))(decoder)
    # scale up to 14 x 14
    decoder = Conv2DTranspose(64, (3, 3), padding='same', activation='relu', strides=2)(decoder)
    # scale up to 28 x 28
    decoder = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid', strides=2)(decoder)
    return decoder


def get_rafae(add_dense=True):
    encoder = get_classifier_model(add_dense)
    decoder = get_decoder_out(encoder.output)
    return Model([encoder.input], [decoder])


def get_cladec(classifier: keras.models.Model, alpha: float, acts_input_shape):
    acts_input = Input(shape=acts_input_shape, dtype=tf.float32)
    # custom inputs for the loss function
    img_input = Input(shape=(28, 28))  # training or test images
    labels_input = Input(shape=(10,))  # categorical encoded true labels
    # decoder
    img_output = get_decoder_out(acts_input)
    model = Model([acts_input, labels_input, img_input], [img_output])
    model.add_loss(loss_cladec_generator(classifier, alpha)(img_input, labels_input, img_output))
    return model
