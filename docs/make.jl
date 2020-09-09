using Pkg; Pkg.activate("..")
push!(LOAD_PATH, "../src/")

using Documenter, Adversarial

makedocs(
    sitename = "Adversarial.jl",
    authors  = "Jay Morgan",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages    = [
        "Home" => "index.md",
        "BlackBox" => "blackbox.md",
        "WhiteBox" => "whitebox.md",
    ]
)

deploydocs(repo = "github.com/jaypmorgan/Adversarial.jl.git")
