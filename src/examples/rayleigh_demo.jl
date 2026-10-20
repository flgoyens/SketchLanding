using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "SketchLanding.jl"))
using .SketchLanding

function random_stiefel(n::Int, p::Int; rng=Random.default_rng())
    Q, _ = qr(randn(rng, n, p))
    return Matrix(Q[:, 1:p])
end

function main()
    rng = MersenneTwister(42)
    n = 20
    p = 2

    # For the Rayleigh trace objective on the Stiefel manifold, use symmetric A.
    Araw = randn(rng, n, n)
    A = 0.5 .* (Araw + Araw')

    gradf! = (G, X) -> begin
        mul!(G, A, X) # G = A*X
        lmul!(2.0, G) # G = 2*G
        nothing
    end

    costf = X -> tr(X' * A * X)

    prob = FunctionProblem(n, p, gradf!; cost_fun=costf)
    X0 = random_stiefel(n, p; rng=rng)

    out = sketchLanding(
        prob,
        X0;
        sketch=:subsample,
        k=15,
        ωN=5.0,
        stepsize=5e-3,
        estimator=:svrg,
        maxiter=2_000,
        rng=rng,
        record=RecordOptions(
            record_infeasibility=true,
            record_gradnorm=true,
            record_norm_iterates=false,
        ),
    )

    final_cost = out.costs[end]
    final_infeas = out.infeasibilities === nothing ? NaN : out.infeasibilities[end]
    final_grad = out.gradnorms === nothing ? NaN : out.gradnorms[end]

    λ = eigvals(Symmetric(A))
    target = sum(sort(λ)[1:p])

    println("SketchLanding Rayleigh example")
    println("n = $n, p = $p")
    println("final cost               = $final_cost")
    println("best eigenvalue sum      = $target")
    println("final infeasibility norm = $final_infeas")
    println("final gradnorm           = $final_grad")
end

main()
