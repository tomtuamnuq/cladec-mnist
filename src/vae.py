import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.models import Sequential, Model

from src.cladec_base import ClaDecBase


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
        self.encoder = self._get_encoder()
        self.decoder = self.create_default_decoder()
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.class_loss_fn = losses.CategoricalCrossentropy()
        self.classifier.trainable = False  # TODO Tensor Board anschauen ?
        self.reconstruction_loss_fn = losses.BinaryCrossentropy()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker
        ]

    def _get_encoder(self):
        code = layers.Flatten(name="code")(self.layer_to_explain.output)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(code)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(code)
        z = Sampling()([z_mean, z_log_var])
        return Model(self.classifier.input, [z_mean, z_log_var, z], name="encoder")

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            y_prime = self.classifier(reconstruction)
            classification_loss = self.class_loss_fn(y, y_prime)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    self.reconstruction_loss_fn(x, reconstruction), axis=(1, 2)
                )  # TODO why not MSE here?
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = (1 - self.alpha) * reconstruction_loss + self.alpha * classification_loss + kl_loss
            # TODO weight kl_loss ? see β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
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
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

# TODO add Reference CLass similar to cladec.py
