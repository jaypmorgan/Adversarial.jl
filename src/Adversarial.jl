module Adversarial

using Flux
using Flux.Tracker: gradient
using Metalhead
using Distances

export FGSM, PGD, JSMA

include("attacks.jl")


end # module Adversarial
