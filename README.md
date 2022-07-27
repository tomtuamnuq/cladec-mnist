# cladec-mnist

An evaluation of the ClaDec architecture to explain layers of convolutional neural networks on the mnist digits and
fashion-mnist dataset.

## Introduction

The paper [Explaining Neural Networks by Decoding Layer Activations](https://arxiv.org/abs/2005.13630) by Johannes
Schneider and Michalis Vlachos introduces the `ClaDec` architecture.
ClaDec explains a layer of a NN by using the NN up to that layer as an encoder, and provides the latent representation
of inputs in that layer as code for a decoder. The decoder then reconstructs inputs based on that code.

Reconstructed inputs are similar to the input domain and, therefore, easy to comprehend. Support is given in
the [extended version of the paper](https://www.semanticscholar.org/paper/Explaining-Classifiers-by-Constructing-Familiar-Schneider-Vlachos/9f8d136595ff962e81a83850612c13ebfeafa115#citing-papers)
by a user study.
To avoid the influence of the decoder part on the inputs recreated, a reference auto encoder with the same architecture
is used in addition. Only the differences between the reconstructions are the actual explanation of the layer.

ClaDec explanations for images are images. The explanation images should contain what the classifier uses to classify.
Concepts or aspects as textures, colors, shapes etc., which are in the ClaDec outputs, should resemble what information
the classifier
maintains and uses to get to a decision. They are important for the classifier. The explanations
are `through the eyes of AI`.
On the other hand, concepts which are not in the ClaDec outputs were not used by the classifier.

Created ClaDec models depend on the `alpha` parameter of the custom loss function. Please see `eq.1`
in [Explaining Neural Networks by Decoding Layer Activations](https://arxiv.org/abs/2005.13630).
The custom loss function is a linear combination between reconstruction and classification loss.
A low value of alpha means that the focus is on reconstruction. The explanations are more input like
with the goal that domain experts are able to derive insights. For a higher value of alpha the focus of the
decoder training is on the "inner life" of the classifier. Explanations should then resemble more
of the internals of the classifier. Thus, `alpha` provides a trade-off between comprehensibility and fidelity.

## Source Code

PyTorch code for the paper on the `VGG-11` architecture is on [GitHub](https://github.com/JohnTailor/ClaDec). It allows
evaluation on `Fashion-MNIST` `CIFAR-10` and `CIFAR-100` datasets. Training the neural networks requires dedicated
hardware.

This repository contains Keras code of an evaluation of `ClaDec` for a small VGG-like classifier architecture on
the `MNIST`
datasets. It is designed to run on a common laptop.

The source code for the creation of the classifier, reference auto encoder and ClaDec are given in `src`. The
script `create_models.py`
trains all models:

A classifier for each of the datasets, a ClaDec decoder for each value of `src.utils.ALPHAS` and each dataset, and a
Reference Auto Encoder for each dataset.
`ClaDec` and RefAE were created two times each. One time to explain a `Dense` layer and one time to
explain a `Convolutional` layer of the classifier.

The `ClaDec` class implementation in `src.cladec.py` can use different decoders for the training of ClaDec.
For the tests I used two dedicated decoders which were designed particularly for the two explained layers. Source code
is in `src.cladec_base.py`.

The architecture for explaining the `Dense` layer is as follows:
![ClaDec_Dense](eval/img/Arch_Example_Cladec.png)
To explain the `Convolutional` layer the `Dense` and first `Convolutional Transpose` layer are dropped.

The reference auto encoder implementation simply copies the encoder and decoder of `ClaDec`. It starts with freshly
initialized weights and is trained on reconstruction loss only:
![ClaDec_Dense](eval/img/Cladec_RefAE.png)

## Results

The final evaluation is given in the Jupyter Notebooks of `eval` dir and the `src.eval.py` script.
Both classifiers (`MNIST` and `Fashion-MNIST`) achieve normal performance (categorical
accuracy `0.9903` and `0.9206`) on unseen data. A qualitative evaluation on reconstructed images is given in
the `Evaluation-Layer` notebooks.
Evidence in data (quantitative evaluation) for total, reconstruction and classification loss for each ClaDec and RefAE
is the output of the `src.eval.py` script (given in `eval.evaluation_result.csv`):
![Eval_Data](eval/img/eval_result_data.jpg)

### Explanations for Fashion MNIST

#### Dense Layer

The following image shows some explanations for `Fashion-MNIST` on the 128 neuron `Dense` layer of the classifier:

![Fashion Dense](eval/img/fashion_mnist/sneaker_dense.png)
The reference auto encoder for `Fashion-MNIST` is not perfect. The reconstructions are a bit blurry and one needs to
compare the output of the ref auto encoder with ClaDec and the original image to derive insights. Some blurriness is the
effect of the chosen decoder architecture. However, the decoder is able to reconstruct images with details and
different grey values.

Explanations from the ClaDec decoder look more blurry and lack details. In fact, it only shows the general outlines
and not the textural details. The classifier seems to only look at the outline and does not rely on grey values.
This would explain why it classifies the sneakers incorrectly as Ankle Boot or Sandal. The classifier cannot distinguish
between sneaker and ankle boot or sandal just by looking at the outline.

Some more examples on the classification of sandals:
![Fashion Dense](eval/img/fashion_mnist/sandal_dense.png)

The classifier seems to try to fit the input into some sort of learned prototype. This is particularly clear for the
wrong classification as sneakers or ankle boots.
It does not capture the necessary details of texture inside the sandals. It just takes the outline (general shape)
and sees a sneaker (or ankle boot).
But there are some contradictions. Why does it classify the first 3 sandals correctly? It seemed to got
at least some details here, which resulted in the darker are in the middle of the sandal. The ClaDec output for those
three looks again prototypically.
The misclassified explanations again lack details of the inside of the shoes. Looking only at the outline makes it
difficult to distinguish between a high sandal and an ankle boot.

### Effects of alpha

The images greatly visualize the effect of alpha. For `alpha=0.0` the training is on only on reconstruction, and it is
easy to compare the explanations with the original ones. For low values of `alpha` the ClaDec outputs look almost
identical.
Sometimes there are darker areas which could highlight what area the classifier finds interesting for prediction.
The idea behind this is that concepts that are important for the classifier should be reconstructed with more detail
than those
that are not important for classification. This is because the higher alpha the more focus is on
classification loss. That means, reconstructions are generated in a way the classifier classifies them correctly.

For `alpha>=0.99` it gets difficult to actually see a shoe in the images. The outputs are formed by small rectangles
with
different
grey values being scattered over the outline of the original shoe. Interestingly the reconstructions of `alpha=0.999` do
not contain
many grey values again. The outputs seem to reflect the effects of striding and pooling in the convolutional layers.

A high value of alpha (e.g. 0.9999) creates strange patterns. It has nothing to do with a shoe, and areas which are not
even used in the original input are turned white or vice versa. However, same classes seem to produce similar patterns.

For my small network the influence of `alpha` on the reconstruction is comparable to the results of Figure 11
in [extended version of the paper](https://www.semanticscholar.org/paper/Explaining-Classifiers-by-Constructing-Familiar-Schneider-Vlachos/9f8d136595ff962e81a83850612c13ebfeafa115#citing-papers)
Even for very high `alpha` reconstructions look quite good.

### Convolutional Layer

Explanations for the `Convolutional` layer:

![Fashion Conv](eval/img/fashion_mnist/sneaker_conv.png)

The decoded images from the last convolutional layer seem to contain way more information than the ones from the dense
layer.
Shoes still contain details such as textures and different grey values. This is the case for the reference AE as well.
From this we could derive that the information loss occurs in the `Dense` layer, or, that the convolutional layers were
not much adjusted for the classification task:
If the classifier would focus on particular learned features in the kernels we should see only those in the
reconstructions. But it could be that the classifier uses only some kernels, and altogether the kernels still have the
information.

For the convolutional layer the influence of alpha is only visible for very high values. One needs a higher value to
produce some effects. Even if the training only uses classification loss (`alpha=1.0`) the form of a shoe is kept.

### Explanations for MNIST

For `MNIST` reconstructions are overall better since the dataset is "easier".
The reference auto encoder was able to reconstruct the `MNIST` images quite well. One therefore only
compares the output of ClaDec with the original images to derive conclusions.

#### Dense Layer

![MNIST Dense](eval/img/mnist/mnist_dense.png)

ClaDec generated images for the correctly classified images are quite good. The classifier seems to be quite certain.
Reconstructions from RefAE are always good.
For the wrongly classified images reconstructions for the classifier are worse which hints to the uncertainty of the
classifier. Blurry regions appear particularly in regions which would be important to distinguish the digits (e.g.
comparing the 9
which was classified as a 7 or the 8 that was classified as a 7).

It does not look as if low values of `alpha` have a visible effect.
For `alpha=0.99` images look a bit different. For the rightmost digits (3 classified as 5 and 5 classified as 3)
the cladec outputs seem to visualize what the classifier actually `sees`: The 3 gets reconstructed as a 5 and vice
versa.
For `alpha=0.999` one sees prototypical digit shapes. The classifier seems to have learned certain prototypical areas.
Such areas could be the most important parts that the classifier uses to distinguish the digits.
For example, there are similar patterns for the images classified as 5 or 3.

For the `Convolutional` layer reconstructions are almost perfect:

![MNIST Conv](eval/img/mnist/mnist_conv.png)

Only for very high `alpha` values some visual effects appear, but they are not any different between correctly and
wrongly classified digits. One is not able to derive insights why the classifications are wrong.

# Variational Decoder

work in progress:

Can ClaDec be extended to derive global insights about the classifier by using a variational decoder?

A first trial gave some prototypical forms from ClaDec:

![Fashion_VAE](eval/img/fashion_mnist/vae_latent_space_cladec.png)

This is to be compared to a reference variational auto encoder:

![Fashion_VAE](eval/img/fashion_mnist/vae_latent_space_ref.png)


