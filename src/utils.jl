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
    p1, p2 = nothing, nothing
    for (p, q) in Base.Iterators.product(CartesianIndices(Γ), CartesianIndices(Γ))
        ((Γ[p] == 1) || (Γ[q] == 1) || (p == q)) && continue  # skip this pixel as its already been modified

        max = 0
        α = j[t,p] + j[t,q]
        β = (sum(j[:,p]) + sum(j[:,q])) - α

        if (α > 0) && (β < 0) && (-α*β > max)
            p1, p2 = p, q
            max = -α * β
        end
    end
    return p1, p2
end


function jacobian(m, x::AbstractArray{T,1}) where T
  y, back = Flux.Tracker.forward(m, x)
  k = size(y,1) # k classes
  J = zeros(T, k, size(x)...)
  ŷ(i) = [i == j for j = 1:k]  # one hot array

  for i in 1:k
    g = back(ŷ(i))[1]
    J[i, :] = g
  end
  return J
end


function jacobian(m, x::AbstractArray{T,4}) where T
  xp = param(x)
  y = m(xp)
  k = size(y,1) # k classes
  J = zeros(T, k, size(x)...)
  ŷ(i) = [i == j for j = 1:k]

  for i in 1:k
    xp = param(x)  # hacky solution... is there a better way of using back! more than once?
    y = m(xp)
    Flux.Tracker.back!(y[i])
    J[i,:,:,:] = copy(xp.grad)
  end
  return J[:,:,:,1]  # don't require the batch dimension
end
