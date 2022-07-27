from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential
from src.utils import get_optimizer, DENSE_LAYER_NAME, CONV_LAYER_NAME


def create_classifier_model_compiled():
    # create VGG like model
    # inspired by https://www.kaggle.com/code/nvsabhilash/keras-vgg-like-architecture-on-mnist-0-98471/notebook
    classifier = Sequential([
        Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'),
        BatchNormalization(), MaxPooling2D(), Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(), MaxPooling2D(name=CONV_LAYER_NAME), Flatten(), Dropout(0.5),
        Dense(128, activation='relu', name=DENSE_LAYER_NAME), Dense(10, activation='sigmoid')])
    classifier.compile(get_optimizer(),
                       'categorical_crossentropy',
                       metrics=['categorical_accuracy'])
    return classifier


def train_classifier(x, y, epochs: int):
    classifier = create_classifier_model_compiled()
    classifier.fit(x, y, epochs=epochs)
    return classifier
