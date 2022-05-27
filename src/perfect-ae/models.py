import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential

from src.common import get_optimizer

tf.random.set_seed(1234)


def get_classifier_model_compiled():
    # create VGG like model
    # inspired by https://www.kaggle.com/code/nvsabhilash/keras-vgg-like-architecture-on-mnist-0-98471/notebook
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    optimizer = get_optimizer()
    model.compile(optimizer, 'categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
