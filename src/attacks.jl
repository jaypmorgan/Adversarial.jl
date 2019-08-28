"""
    FGSM(model, loss, x, y; ϵ = 0.1, clamp_range = (0, 1))

Fast Gradient Sign Method (FGSM) is a method of creating adversarial examples
by pushing the input in the direction of the gradient and bounded by the ε parameter.

This method was proposed by Goodfellow et al. 2014 (https://arxiv.org/abs/1412.6572)

## Arguments:
- `model`: The model to base the attack upon.
- `loss`: The loss function to use. This assumes that the loss function includes
    the predict function, i.e. loss(x, y) = crossentropy(model(x), y).
- `x`: The input to be perturbed by the FGSM algorithm.
- `y`: The 'true' label of the input.
- `ϵ`: The amount of perturbation to apply.
- `clamp_range`: Tuple consisting of the lower and upper values to clamp the input.
"""
function FGSM(model, loss, x, y; ϵ = 0.1, clamp_range = (0, 1))
    x, θ = param(x), params(model)
    J = gradient(() -> loss(x, y), θ)
    x_adv = clamp.(x.data + (ϵ * sign.(x.grad)), clamp_range...)
    return x_adv
end


"""
    PGD(model, loss, x, y; ϵ = 10, step_size = 0.1, iters = 100, clamp_range = (0, 1))

Projected Gradient Descent (PGD) is an itrative variant of FGSM with a random
point. For every step the FGSM algorithm moves the input in the direction of
the gradient bounded in the l∞ norm.
(https://arxiv.org/pdf/1706.06083.pdf)

## Arguments:
- `model`: The model to base teh attack upon.
- `loss`: the loss function to use, assuming that it includes the prediction function
    i.e. loss(x, y) = crossentropy(m(x), y)
- `x`: The input to be perturbed.
- `y`: the ground truth for x.
- `ϵ`: The bound around x.
- `step_size`: The ϵ value in the FGSM step.
- `iters`: The maximum number of iterations to run the algorithm for.
- `clamp_range`: The lower and upper values to clamp the input to.
"""
function PGD(model, loss, x, y; ϵ = 10, step_size = 0.1, iters = 100, clamp_range = (0, 1))
    x_adv = x + (randn(size(x)...) * step_size); # start from the random point
    δ = chebyshev(x, x_adv)
    while (δ < ϵ) && iter <= iteres
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        δ = chebyshev(x, x_adv)
        iter += 1
    end
    return x_adv
end


"""

"""
function JSMA()

end


"""

"""
function CW()

end
