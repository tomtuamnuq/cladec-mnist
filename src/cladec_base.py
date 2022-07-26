import abc

import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.models import Sequential, Model


class ClaDecBase(keras.Model, abc.ABC):

    def __init__(self, classifier: Sequential, layer_name: str, alpha: float, **kwargs):
        super(ClaDecBase, self).__init__(**kwargs)
        self.alpha = alpha
        self.layer_to_explain = classifier.get_layer(layer_name)
        self.classifier = classifier
        self.latent_dim = None
        self.encoder = None
        self.decoder = None
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")

    @tf.function
    def classification_loss_fn(self, x, y):
        y_prime = self.classifier(x)
        return tf.reduce_mean(self.classifier.compiled_loss(y, y_prime))

    @staticmethod
    @tf.function
    def reconstruction_loss_fn(x, y, axis=(1, 2)):
        """Mean of squared distance over the batch. Default axis for image data."""
        return tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=axis))

    @abc.abstractmethod
    def call(self, inputs):
        return NotImplemented

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.class_loss_tracker]

    @abc.abstractmethod
    def create_encoder(self):
        return NotImplemented

    def create_default_decoder(self):
        # default encoder for MNIST and Fashion-MNIST
        latent_inputs = keras.Input(shape=self.latent_dim)
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        return Model(latent_inputs, decoder_outputs, name="decoder")

    @staticmethod
    def create_128_dense_decoder():
        latent_inputs = keras.Input(shape=128)
        decoder = layers.Reshape((1, 1, 128))(latent_inputs)
        decoder = layers.Conv2DTranspose(64,
                                         (4, 4),
                                         padding='valid',
                                         strides=(2, 2),
                                         activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(32,
                                         (3, 3),
                                         padding='valid',
                                         strides=(2, 2),
                                         activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(1,
                                         (3, 3),
                                         padding='valid',
                                         strides=(3, 3),
                                         activation='sigmoid',
                                         output_padding=1)(decoder)

        return Model(latent_inputs, decoder, name="decoder")

    @staticmethod
    def create_64_conv_decoder():
        latent_inputs = keras.Input(shape=3136)
        decoder = layers.Reshape((7, 7, 64))(latent_inputs)
        decoder = layers.Conv2DTranspose(64,
                                         (4, 4),
                                         padding='same',
                                         strides=(2, 2),
                                         activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(1,
                                         (4, 4),
                                         padding='same',
                                         strides=(2, 2),
                                         activation='sigmoid')(decoder)
        return Model(latent_inputs, decoder, name="decoder")

    @abc.abstractmethod
    def train_step(self, data):
        return NotImplemented

    @abc.abstractmethod
    def test_step(self, data):
        return NotImplemented
