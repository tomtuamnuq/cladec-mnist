# cladec-mnist
An evaluation of the ClaDec architecture to explain layers of convolutional neural networks on the mnist dataset.

The paper [Explaining Neural Networks by Decoding Layer Activations](https://arxiv.org/abs/2005.13630) by Johannes Schneider and Michalis Vlachos introduces the `ClaDec` architecture.
ClaDec explains a layer of a NN by using the NN up to that layer as an encoder and provide the latent representation of inputs in that layer as code for a decoder. The decoder then reconstructs inputs based on that code.
Reconstructed inputs are similar to the input domain and therefore easy to comprehend. Support is given in the [extended version of the paper](https://www.semanticscholar.org/paper/Explaining-Classifiers-by-Constructing-Familiar-Schneider-Vlachos/9f8d136595ff962e81a83850612c13ebfeafa115#citing-papers) by a user study.
To avoid the influence of the decoder part on the inputs recreated, a reference auto encoder with the same architecture is used in addition. Only the differences between the reconstructions are the actual explanation of the layer.

PyTorch code for the paper on the `VGG-11` architecture is on [GitHub](https://github.com/JohnTailor/ClaDec). It allows evaluation on `Fashion-MNIST` `CIFAR-10` and `CIFAR-100` datasets. Training the neural networks requires dedicated hardware.

This repository contains Keras code of an evaluation of `ClaDec` for two small VGG-like architectures on the `MNIST` dataset. It is designed to run on a common laptop.
