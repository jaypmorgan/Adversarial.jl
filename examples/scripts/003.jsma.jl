#' # Using JSMA To Create Adversarial Examples
#'
#'

#'
using Adversarial
using Flux
using Statistics
using Flux.Tracker: gradient, update!
using Flux.Data.MNIST
using CuArrays # comment out if a GPU is not available
using Images
using Plots

#' next we will load the trianing. For the example, we will only perform adversarial
#' attacks on this data
train_images = MNIST.images()
train_labels = MNIST.labels()

#' A function to conver the images to arrays
function minibatch(i, batch_size = 32)
    x_batch = Array{Float32}(undef, size(train_images[1])..., 1, batch_size)
    y_batch = Flux.onehotbatch(train_labels[i:i+batch_size-1], 0:9)
    for (idx, image) in enumerate(train_images[i:i+batch_size-1])
        x_batch[:,:,1,idx] = Float32.(image)
    end
    return x_batch |> gpu, y_batch |> gpu
end

#' We then create and train a simple CNN.
CNN() = Chain(
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10),
    softmax) |> gpu

m = CNN();
θ = params(m);
loss(x, y) = Flux.crossentropy(m(x), y)
acc(x, y) = mean(Flux.onecold(m(x)) |> cpu .== Flux.onecold(y) |> cpu)
opt = ADAM()

#' now for our training loop...
const EPOCHS = 5
const BATCH_SIZE = 32

for epoch in 1:EPOCHS
    losses = 0.
    accuracies = 0.
    steps = 0

    for i in 1:BATCH_SIZE:size(train_images,1)-BATCH_SIZE
        x, y = minibatch(i)
        l = loss(x, y)
        a = acc(x, y)

        g = gradient(() -> l, θ)
        for p in θ
            update!(opt, p, g[p])
        end

        losses += l |> Tracker.data
        accuracies += a
        steps += 1
    end
    @info "Epoch $epoch end. Loss: $(losses / steps), Acc: $(accuracies / steps)"
end

#'
x, y = minibatch(1, 1)
y |> Flux.onecold |> getindex
x_adv = JSMA(m, x, 9; Υ = 50, θ = 0.5)


#' we can see that the predicted labels are different
adversarial_pred = m(x_adv) |> Flux.onecold |> getindex
original_pred = m(x) |> Flux.onecold |> getindex

#' and visualise the resulting adversarial in comparison to the original image.
#' When using an ϵ value of 0.07, the different is very slight, if noticable
#' at all.
l = @layout [a b]
adv = heatmap(permutedims(x_adv, (4, 3, 1, 2))[1,1,:,:] |> cpu)
org = heatmap(permutedims(x, (4, 3, 1, 2))[1,1,:,:] |> cpu)
plot(org, adv, layout = l)

@assert adversarial_pred != original_pred
