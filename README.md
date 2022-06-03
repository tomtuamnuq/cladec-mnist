# cladec-mnist

An evaluation of the ClaDec architecture to explain layers of convolutional neural networks on the mnist digits and
fashion-mnist dataset.

The paper [Explaining Neural Networks by Decoding Layer Activations](https://arxiv.org/abs/2005.13630) by Johannes
Schneider and Michalis Vlachos introduces the `ClaDec` architecture.
ClaDec explains a layer of a NN by using the NN up to that layer as an encoder and provide the latent representation of
inputs in that layer as code for a decoder. The decoder then reconstructs inputs based on that code.
Reconstructed inputs are similar to the input domain and therefore easy to comprehend. Support is given in
the [extended version of the paper](https://www.semanticscholar.org/paper/Explaining-Classifiers-by-Constructing-Familiar-Schneider-Vlachos/9f8d136595ff962e81a83850612c13ebfeafa115#citing-papers)
by a user study.
To avoid the influence of the decoder part on the inputs recreated, a reference auto encoder with the same architecture
is used in addition. Only the differences between the reconstructions are the actual explanation of the layer.

PyTorch code for the paper on the `VGG-11` architecture is on [GitHub](https://github.com/JohnTailor/ClaDec). It allows
evaluation on `Fashion-MNIST` `CIFAR-10` and `CIFAR-100` datasets. Training the neural networks requires dedicated
hardware.

This repository contains Keras code of an evaluation of `ClaDec` for two small VGG-like architectures on the `MNIST`
datasets. It is designed to run on a common laptop.

The source code for the creation of the classifier, reference auto encoder and ClaDec are given in `src`. There are two
versions of neural networks for the `MNIST` digits dataset:
`perfectAE` has an architecture with more layers and neurons than `diffAE`. The model building is shown in the Jupyter
Notebooks of `create-models` dir.
The used reference auto encoder in `perfectAE` is able to reconstruct the `MNIST` images quite well. One therefore only
compares the output of ClaDec with the original images to derive explanations.
The reference auto encoder in `diffAE` is not that good. The reconstructions are more blurry and one needs to compare
the output of the ref auto encoder with ClaDec and the original image.
Both classifiers (`perfectAE` and `diffAE`) achieve good performance on the `MNIST` test data (categorical
accuracy `0.9899` and `0.9863`). For `Fashion-MNIST` only the `perfectAE` architecture was applied.

`ClaDec` was created two times each. One time to explain a `Dense` layer and one time to
explain a `Convolutional` layer.
Created ClaDec models depend on the `alpha` parameter of the loss function. Please see `eq.1`
in [Explaining Neural Networks by Decoding Layer Activations](https://arxiv.org/abs/2005.13630).
For `diffAE` another parameter `percentage of layer activations` is used here to explain a subset of neurons in the
layer (compare to `Section 6.4`
in [extended version of the paper](https://www.semanticscholar.org/paper/Explaining-Classifiers-by-Constructing-Familiar-Schneider-Vlachos/9f8d136595ff962e81a83850612c13ebfeafa115#citing-papers))
.

The final evaluation is given in the Jupyter Notebooks of `eval` dir. To be continued...




