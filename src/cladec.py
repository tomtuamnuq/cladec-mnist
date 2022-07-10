import pathlib

import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.models import Sequential, Model, clone_model

from src.cladec_base import ClaDecBase


class ClaDec(ClaDecBase):

    def __init__(self, classifier: Sequential, layer_name: str, alpha: float, decoder: Model = None, **kwargs):
        super(ClaDec, self).__init__(classifier, layer_name, alpha, **kwargs)
        self.encoder = self.create_encoder()
        if decoder is None:
            self.decoder = self.create_default_decoder()
        else:
            self.decoder = decoder
        self.classifier.trainable = False

    @classmethod
    def create_from_weights(cls, path_to_weights: pathlib.Path, classifier: Sequential, layer_name: str, alpha: float,
                            decoder: Model = None, **kwargs):
        cladec = ClaDec(classifier, layer_name, alpha, decoder, **kwargs)
        cladec.load_weights(path_to_weights)
        cladec.compile()
        return cladec

    def create_encoder(self):
        code = layers.Flatten(name="code")(self.layer_to_explain.output)
        self.latent_dim = code.shape[-1]
        return Model(self.classifier.input, code)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            classification_loss, reconstruction_loss, total_loss = self._calc_losses(x, y)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self.update_loss_tracker(classification_loss, reconstruction_loss, total_loss)

    @tf.function
    def _calc_losses(self, x, y):
        reconstruction = self(x, training=True)
        classification_loss = self.classification_loss_fn(reconstruction, y)
        reconstruction_loss = self.reconstruction_loss_fn(x, reconstruction)
        total_loss = (1 - self.alpha) * reconstruction_loss + self.alpha * classification_loss
        return classification_loss, reconstruction_loss, total_loss

    @tf.function
    def test_step(self, data):
        x, y = data
        classification_loss, reconstruction_loss, total_loss = self._calc_losses(x, y)
        return self.update_loss_tracker(classification_loss, reconstruction_loss, total_loss)

    @tf.function
    def update_loss_tracker(self, classification_loss, reconstruction_loss, total_loss):
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.class_loss_tracker.update_state(classification_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classification_loss": self.class_loss_tracker.result()
        }

    @tf.function
    def call(self, inputs):
        code = self.encoder(inputs)
        return self.decoder(code)


class RefAE(keras.Model):

    def __init__(self, claDec: ClaDec, **kwargs):
        super(RefAE, self).__init__(**kwargs)
        self.encoder = clone_model(claDec.encoder)
        self.decoder = clone_model(claDec.decoder)
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.reconstruction_loss_fn = claDec.reconstruction_loss_fn
        self.encoder.trainable = True

    @classmethod
    def create_from_weights(cls, path_to_weights: pathlib.Path, claDec: ClaDec, **kwargs):
        refae = RefAE(claDec, **kwargs)
        refae.load_weights(path_to_weights)
        refae.compile()
        return refae

    @tf.function
    def call(self, inputs):
        code = self.encoder(inputs)
        return self.decoder(code)

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction = self(data, training=True)
            reconstruction_loss = self.reconstruction_loss_fn(data, reconstruction)
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        reconstruction = self(data, training=False)
        reconstruction_loss = self.reconstruction_loss_fn(data, reconstruction)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
