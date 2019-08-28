"""
    FGSM(loss, input, target; ϵ = 0.1)

Fast Gradient Sign Method (FGSM) is a method of creating adversarial examples
by pushing the input in the direction of the gradient and bounded by the ε parameter.

This method was proposed by Goodfellow et al. 2014 (https://arxiv.org/abs/1412.6572)

## Arguments:
- `loss`: The loss function to use. This assumes that the loss function includes
    the predict function, i.e. loss(x, y) = crossentropy(model(x), y).
- `input`: The input to be perturbed by the FGSM algorithm.
- `target`: The 'true' label of the input.
"""
function FGSM(model, loss, input, target; ε = 0.1)
    x, θ = param(input), params(model)
    J = gradient(() -> loss(x, target), θ)
    η = input + (ε * sign.(x.grad))
    return η
end


"""


"""
function PGD()

end


"""

"""
function JSMA()

end


"""

"""
function CW()

end
