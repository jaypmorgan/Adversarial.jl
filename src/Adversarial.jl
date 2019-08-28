module Adversarial

using Flux
using Flux.Tracker: gradient
using Distances

export FGSM, PGD, JSMA

include("attacks.jl")


end # module Adversarial
