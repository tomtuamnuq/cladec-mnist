import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, Model

from src.cladec_base import ClaDecBase


class ClaDec(ClaDecBase):

    def __init__(self, classifier: Sequential, layer_name: str, alpha: float, **kwargs):
        super(ClaDec, self).__init__(classifier, layer_name, alpha, **kwargs)
        code = layers.Flatten()(self.layer_to_explain.output)
        self.encoder = Model(self.layer_to_explain.output, code)
        self.latent_dim = code.shape[-1]
        self.decoder = self._get_decoder()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.class_loss_tracker
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            acts = self.model_up_to_layer_to_explain(x)
            y_prime = self.classifier(acts)
            classification_loss = self.class_loss_fn(y, y_prime)
            z = self.encoder(acts)
            reconstruction = self.decoder(z)
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
