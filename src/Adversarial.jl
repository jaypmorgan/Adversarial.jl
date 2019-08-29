module Adversarial

using Optim
using Flux
using Flux.Tracker: gradient
using Distances

export FGSM, PGD, JSMA

include("utils.jl")
include("attacks.jl")


end # module Adversarial
