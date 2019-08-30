"""
    Loss term used in the C&W paper
"""
function f6_loss(δ, x, t, model, dist, c)
    diff = clamp.(x .+ δ, 0, 1)
    Z = model(diff)
    target_pred = Z[t]
    untargeted_pred = -Inf
    for (i, pred) in enumerate(Z)
        if (i != t) && (pred > untargeted_pred)
            untargeted_pred = pred
        end
    end
    dist(diff, x) + (c * max(untargeted_pred - target_pred, Float32(0)))
end
