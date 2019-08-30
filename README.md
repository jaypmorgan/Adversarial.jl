# Adversarial.jl

Adversarial attacks for Neural Networks written with FluxML.

Adversarial examples are inputs to Neural Networks (NNs) that result in a miss-classification. Through the exploitation that NNs are susceptible to slight changes (perturbations) of the input space, adversarial examples are often indistinguishable from the original input from the point of view of a human.

A common example of this phenomenon is from the use of the Fast Gradient Sign Method (FGSM) proposed by Goodfellow et al. 2014 https://arxiv.org/abs/1412.6572 where the gradient information of the network can be used to move the pixels of the image in the direction of gradient, and thereby increasing the loss for the resulting image. Despite the very small shift in pixels, this is enough for the NN to miss-classify the image: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

We have included some of the common methods to create adversarial examples, this includes:

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Jacobian-based Saliency Map Attack (JSMA)
- Carlini & Wagner (CW)
