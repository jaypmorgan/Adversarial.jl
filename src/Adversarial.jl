module Adversarial

using Random
using Flux
using Flux.Tracker: gradient
using Distances

export FGSM, PGD, JSMA, CW

include("loss.jl")
include("utils.jl")
include("whitebox.jl")
include("blackbox.jl")

end # module Adversarial
