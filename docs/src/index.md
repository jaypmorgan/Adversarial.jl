# Adversarial.jl Documentation


Adversarial attacks for Neural Networks written with FluxML.

Adversarial examples are inputs to Neural Networks (NNs) that result in a miss-classification. Through the exploitation that NNs are susceptible to slight changes (perturbations) of the input space, adversarial examples are often indistinguishable from the original input from the point of view of a human.

A common example of this phenomenon is from the use of the Fast Gradient Sign Method (FGSM) proposed by Goodfellow et al. 2014 https://arxiv.org/abs/1412.6572 where the gradient information of the network can be used to move the pixels of the image in the direction of gradient, and thereby increasing the loss for the resulting image. Despite the very small shift in pixels, this is enough for the NN to miss-classify the image: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

We have included some of the common methods to create adversarial examples, this includes:

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Jacobian-based Saliency Map Attack (JSMA)
- Carlini & Wagner (CW)

## Installation

You can install this package through Julia's package manager in the REPL:

```julia
] add Adversarial
```

or via a script:

```julia
using Pkg; Pkg.add("Adversarial")
```

## Quick Start Guide

As an example, we can create an adversarial image using the FGSM method:

```julia
x_adv = FGSM(model, loss, x, y; ϵ = 0.07)
```

Where model is the FluxML model, loss is some loss function that uses a predict function, for example `crossentropy(model(x), y)`. x is the original input, y is the true class label, and \epsilon is a parameter that determines how much each pixel is changed by.

## Index

```@index
```
