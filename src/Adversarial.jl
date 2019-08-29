module Adversarial

using Optim
using Flux
using Flux.Tracker: gradient
using Distances

export FGSM, PGD, JSMA, CW

include("loss.jl")
include("utils.jl")
include("attacks.jl")

end # module Adversarial
