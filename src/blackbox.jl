"""

(https://arxiv.org/pdf/1905.07121.pdf)

## Arguments:
- `model`: The flux model to attack.
- `x`: Original input data to perturb
- `y`: The ground truth index for x.
- 'ϵ': The maximum amount of perturbation per dimension.
"""
function SimBA(model, x, y::Int; ϵ = 0.1)
    δ = 0
    Q = randperm(collect(CartesianIndices(x)))
    p = model(x)[y]
    max_y = model(x) |> Flux.onecold |> getindex
    counter = 0
    while p == max_y
        q = x[Q[counter]]
        for α in [ϵ,-ϵ]
            p_prime = model(x .+ δ + αq)[y]
            if p_prime < p
                δ = δ + αq
                p = copy(p_prime)
                break
            end
        end
        counter += 1
        max_y = model(x .+ δ) |> Flux.onecold |> getindex
    end
    return (x .+ δ)
end
