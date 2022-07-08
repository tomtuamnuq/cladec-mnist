import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.models import Sequential, Model, clone_model

from src.cladec_base import ClaDecBase


class ClaDec(ClaDecBase):

    def __init__(self, classifier: Sequential, layer_name: str, alpha: float, decoder: Model = None, **kwargs):
        super(ClaDec, self).__init__(classifier, layer_name, alpha, **kwargs)
        self.encoder = self._get_encoder()
        if decoder is None:
            self.decoder = self.create_default_decoder()
        else:
            self.decoder = decoder
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.class_loss_fn = losses.CategoricalCrossentropy()  # TODO get from classifier
        self.classifier.trainable = False  # TODO Tensor Board anschauen ?

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.class_loss_tracker
        ]

    def _get_encoder(self):
        code = layers.Flatten(name="code")(self.layer_to_explain.output)
        self.latent_dim = code.shape[-1]
        return Model(self.classifier.input, code)

    @tf.function
    def train_step(self, data):  # TODO eager mode - Python just in time, anderer Modus graph_execution Kompilierung
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction = self(x)
            y_prime = self.classifier(reconstruction)
            classification_loss = self.class_loss_fn(y, y_prime)
            reconstruction_loss = self.reconstruction_loss_fn(x, reconstruction)
            total_loss = (1 - self.alpha) * reconstruction_loss + self.alpha * classification_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
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

    @tf.function
    def call(self, inputs):
        code = self.encoder(inputs)
        return self.decoder(code)

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction = self(data)
            reconstruction_loss = self.reconstruction_loss_fn(data, reconstruction)
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
