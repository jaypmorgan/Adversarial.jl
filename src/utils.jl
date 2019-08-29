function jacobian(m, x::AbstractArray{T,1}) where T
  y, back = Flux.Tracker.forward(m, x)
  k = size(y,1) # k classes
  J = zeros(T, k, size(x)...)
  ŷ(i) = [i == j for j = 1:k]  # one hot array

  for i in 1:k
    g = back(ŷ(i))[1]
    J[i, :] = g |> Tracker.data
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
