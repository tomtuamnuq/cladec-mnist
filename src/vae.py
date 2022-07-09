import pathlib

import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.models import Sequential, Model, clone_model

from src.cladec_base import ClaDecBase


# inspired by https://keras.io/examples/generative/vae/
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding."""

    @tf.function
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ClaDecVAE(ClaDecBase):

    def __init__(self, classifier: Sequential, layer_name: str, alpha: float, latent_dim: int = 2, **kwargs):
        super(ClaDecVAE, self).__init__(classifier, layer_name, alpha, **kwargs)
        self.latent_dim = latent_dim
        self.encoder = self.create_encoder()
        self.decoder = self.create_default_decoder()
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.class_loss_fn = losses.CategoricalCrossentropy()
        self.classifier.trainable = False

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker
        ]

    def create_encoder(self):
        code = layers.Flatten(name="code")(self.layer_to_explain.output)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(code)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(code)
        z = Sampling()([z_mean, z_log_var])
        return Model(self.classifier.input, [z_mean, z_log_var, z], name="encoder")

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            classification_loss, kl_loss, reconstruction_loss, total_loss = self._calc_losses(x, y)
            # TODO weight kl_loss ? see β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self.update_loss_tracker(classification_loss, kl_loss, reconstruction_loss, total_loss)

    @tf.function
    def update_loss_tracker(self, classification_loss, kl_loss, reconstruction_loss, total_loss):
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(classification_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "classification_loss": self.class_loss_tracker.result()
        }

    @tf.function
    def test_step(self, data):
        x, y = data
        classification_loss, kl_loss, reconstruction_loss, total_loss = self._calc_losses(x, y)
        return self.update_loss_tracker(classification_loss, kl_loss, reconstruction_loss, total_loss)

    @tf.function
    def _calc_losses(self, x, y):
        z_mean, z_log_var, reconstruction = self(x)
        y_prime = self.classifier(reconstruction)
        classification_loss = self.class_loss_fn(y, y_prime)
        reconstruction_loss = self.reconstruction_loss_fn(x, reconstruction)
        kl_loss = self.kl_loss_fn(z_mean, z_log_var)
        total_loss = (1 - self.alpha) * reconstruction_loss + self.alpha * classification_loss + kl_loss
        return classification_loss, kl_loss, reconstruction_loss, total_loss

    @staticmethod
    @tf.function
    def reconstruction_loss_fn(x, reconstruction):
        return tf.reduce_mean(
            tf.reduce_sum(
                losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
            )
        )

    @staticmethod
    @tf.function
    def kl_loss_fn(z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return kl_loss

    @tf.function
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction


class RefVAE(keras.Model):

    def __init__(self, claDec: ClaDecVAE, **kwargs):
        super(RefVAE, self).__init__(**kwargs)
        self.encoder = clone_model(claDec.encoder)
        self.decoder = clone_model(claDec.decoder)
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.encoder.trainable = True

    @classmethod
    def create_from_weights(cls, path_to_weights: pathlib.Path, claDec: ClaDecVAE, **kwargs):
        refae = RefVAE(claDec, **kwargs)
        refae.load_weights(path_to_weights)
        refae.compile()
        return refae

    @tf.function
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    @tf.function
    def _calc_losses(self, x):
        z_mean, z_log_var, reconstruction = self(x)
        reconstruction_loss = ClaDecVAE.reconstruction_loss_fn(x, reconstruction)
        kl_loss = ClaDecVAE.kl_loss_fn(z_mean, z_log_var)
        total_loss = reconstruction_loss + kl_loss
        return kl_loss, reconstruction_loss, total_loss

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            kl_loss, reconstruction_loss, total_loss = self._calc_losses(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self.update_loss_tracker(kl_loss, reconstruction_loss, total_loss)

    @tf.function
    def update_loss_tracker(self, kl_loss, reconstruction_loss, total_loss):
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        kl_loss, reconstruction_loss, total_loss = self._calc_losses(data)
        return self.update_loss_tracker(kl_loss, reconstruction_loss, total_loss)
