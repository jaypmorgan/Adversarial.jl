# internal
using Test
using Random; Random.seed!(42);

# External Packages
using Flux
using Flux.Data: MNIST

# Test Package
using Adversarial

# create the NNs for tests
model() = Chain(Conv((5,5), 1=>10, relu),
                Conv((5,5), 10=>20, relu),
                Flux.flatten,
                Dense(8000, 50, relu),
                Dense(50, 10),
                softmax)
m = model();

# helper functions
loss(x, y) = Flux.crossentropy(m(x), y)
f(xi) = m(xi) |> Flux.onecold |> getindex

# take one image from the validation dataset
sample_n = 1
x = Float32.(MNIST.images(:test)[sample_n])
x = reshape(x, size(x,1), size(x,1), 1, 1)
y = Flux.onehot(MNIST.labels(:test)[sample_n], 1:10)


# run tests
# =============================================================================-
@test f(FGSM(m, loss, x, y)) != f(x)
