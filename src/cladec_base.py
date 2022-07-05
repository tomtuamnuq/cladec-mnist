import abc

from tensorflow import keras
from keras import layers, losses
from keras.models import Sequential, Model


class ClaDecBase(keras.Model, abc.ABC):

    def __init__(self, classifier: Sequential, layer_name: str, alpha: float, **kwargs):
        super(ClaDecBase, self).__init__(**kwargs)
        self.alpha = alpha
        self.layer_to_explain = classifier.get_layer(layer_name)
        self.model_up_to_layer_to_explain = Model(classifier.input, self.layer_to_explain.output)
        self.classifier = Model(self.model_up_to_layer_to_explain.output, classifier.output, name="classifier")
        self.latent_dim = None
        self.encoder = None
        self.decoder = None
        classifier.trainable = False
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.reconstruction_loss_fn = losses.MeanSquaredError()
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.class_loss_fn = losses.CategoricalCrossentropy()

    def call(self, inputs):
        acts = self.model_up_to_layer_to_explain(inputs)
        z = self.encoder(acts)
        return self.decoder(z)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.class_loss_tracker
        ]

    def _get_decoder(self):
        # default encoder for MNIST and Fashion-MNIST
        latent_inputs = keras.Input(shape=self.latent_dim)
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        return Model(latent_inputs, decoder_outputs, name="decoder")

    @abc.abstractmethod
    def train_step(self, data):
        return NotImplemented
