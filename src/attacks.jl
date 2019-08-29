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
    while (δ < ϵ) && iter <= iters
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        δ = chebyshev(x, x_adv)
        iter += 1
    end
    return x_adv
end


"""
    JSMA(model, x, t; Υ, θ)

Jacobian Saliency Map Algorithm (JSMA), craft adversarial examples
by modifying a very small amount of pixels. These pixels are selected
via the jacobian matrix of the output w.r.t. the input of the network.
(https://arxiv.org/pdf/1511.07528.pdf)

## Arguments:
- `model`: The model to create adversarial examples for.
- `x`: The original input data
- `t`: Index corrosponding to the target class (this is a targeted attack).
- `Υ`: The maximum amount of distortion
- `θ`: The amount by which each feature is perturbed.

"""
function JSMA(model, x, t; Υ, θ, clamp_range = (0, 1))
    label(in) = model(in) |> Flux.onecold |> getindex

    x_adv = copy(x)
    Γ = zeros(size(x,1), size(x,2))
    max_iter = (Γ * Υ) / (2 * 100)
    s = label(x)
    iter = 0

    while (s != t) && iter < max_iter && !(all(Γ))
        p1, p2 = saliency_map(jacobian(model, x_adv), Γ, t)
        Γ[p1], Γ[p2] = 1, 1 # set these indexes to modified
        x_adv[p1...,:] = clamp.(x_adv[p1...,:] + θ, clamp_range...)
        x_adv[p2...,:] = clamp.(x_adv[p2...,:] + θ, clamp_range...)
        s = label(x_adv)
        iter += 1
    end
    return x_adv
end


"""
    saliency_map(j, Γ, t)

Determine the optimal pixels to change based upon the saliency via the
jacobian. This method is used as part of the JSMA algorithm. It returns
the cartesian index of the best pixels to modify.

## Arguments:
- `j`: The jacobian matrix of outputs w.r.t. inputs
- `Γ`: The matrix of pixels where 0 denotes that the pixel has yet to be
    modified. I.e. the search space
- `t`: Target class index.
"""
function saliency_map(j, Γ, t)
    for (p, q) in Base.Iterators.product(CartesianIndices(Γ), CartesianIndices(Γ))
        ((Γ[p] == 1) || (Γ[q] == 1)) && (p == q) && continue  # skip this pixel as its already been modified

        max = 0
        α = j[p..., t] + j[q..., t]
        β = (sum(j[p..., :]) + sum(j[q..., :])) - α

        if (α > 0) && (β < 0) && (-α*β > max)
            p1, p2 = p, q
            max = -α * β
        end
    end
    return p1, p2
end


"""
    CW(model, x, t; dist = euclidean, c = 0.1)

Carlini & Wagner's (CW) method for generating adversarials through the optimisation
of a loss function against a target class. Here we consider the F6 variant loss
function. (https://arxiv.org/pdf/1608.04644.pdf)

## Arguments:
- `model`: The model to attack.
- `x`: The original input data
- `t`: Index label corrosponding to the target class.
- `dist`: The distance measure to use L0, L2, L∞. Assumes this is from the
    Distances.jl library or some other callable function.
- `c`: value for the contribution of the missclassification in the error function.
"""
function CW(model, x, t::Int; dist::Function = euclidean, c = 0.1)
    f6_loss(x_adv) = begin
        δ = max.(min.(x_adv, 1), 0)
        f = max(max(model(δ)[1:end .!= t]) .- model(δ)[t], 0)
        dist(δ, x) + c * f
    end
    x_adv = copy(x)
    x_adv = optimize(f6_loss, x_adv, GradientDescent(); autodiff = :forward).minimizer
end
