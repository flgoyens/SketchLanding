using Test
using Random
using LinearAlgebra
using SparseArrays

using SketchLanding

function toy_problem(nv, pv)
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    return SketchLanding.FunctionProblem(nv, pv, gradf!)
end

function normal_reference(sk1, sk2, X)
    n = size(X, 1)
    B1 = SketchLanding.dense_B(sk1, n)
    B2 = SketchLanding.dense_B(sk2, n)
    return (B2 * X) * (X' * (B1 * X) - I)
end

function tangent_reference(sk1, sk2, X, G)
    n = size(X, 1)
    B1 = SketchLanding.dense_B(sk1, n)
    B2 = SketchLanding.dense_B(sk2, n)
    Kref = X' * (B1 * (B2 * X))
    Pref = G' * (B2 * X)
    return G * Kref - (B1 * X) * Pref
end

@testset "SX applications" begin
    rng = MersenneTwister(7)
    n, p, k = 9, 3, 4
    X = randn(rng, n, p)

    skI = SketchLanding.IdentitySketch()
    YI = zeros(n, p)
    SketchLanding.StX!(YI, skI, X)
    @test YI ≈ X

    skG = SketchLanding.GaussianSketch(n, k)
    sample!(skG, rng)
    YG = zeros(k, p)
    SketchLanding.StX!(YG, skG, X)
    @test YG ≈ skG.S' * X

    skS = SketchLanding.SubsampleSketch(n, k)
    skS.idx .= [1, 3, 6, 8]
    YS = zeros(k, p)
    SketchLanding.StX!(YS, skS, X)
    @test YS ≈ skS.c .* X[skS.idx, :]
end

@testset "Subsample BX and XtBX" begin
    n, p, k = 8, 3, 3
    X = reshape(collect(1.0:n*p), n, p)
    sk = SketchLanding.SubsampleSketch(n, k)
    sk.idx .= [2, 4, 7]

    Y = zeros(n, p)
    SketchLanding.BX!(Y, sk, X)

    B = zeros(n, n)
    c2 = sk.c^2
    for i in sk.idx
        B[i, i] = c2
    end
    @test Y ≈ B * X

    M = zeros(p, p)
    SketchLanding.XtBX!(M, sk, X)
    @test M ≈ X' * B * X
end

@testset "Intersection positions" begin
    I = [1, 3, 5, 7, 9]
    J = [2, 3, 4, 5, 10]
    posI = zeros(Int, 5)
    posJ = zeros(Int, 5)
    r = SketchLanding.intersect_positions!(posI, posJ, I, J)

    @test r == 2
    @test posI[1:r] == [2, 3]
    @test posJ[1:r] == [2, 4]
end

@testset "XtB1B2X subsampling specialization" begin
    n, p, k = 7, 2, 3
    X = randn(n, p)
    sk1 = SketchLanding.SubsampleSketch(n, k)
    sk2 = SketchLanding.SubsampleSketch(n, k)
    sk1.idx .= [1, 3, 5]
    sk2.idx .= [2, 3, 5]

    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :subsample, :plain; k=k)

    K = zeros(p, p)
    SketchLanding.XtB1B2X!(K, sk1, sk2, X, st)

    B1 = zeros(n, n)
    B2 = zeros(n, n)
    for i in sk1.idx
        B1[i, i] = sk1.c^2
    end
    for i in sk2.idx
        B2[i, i] = sk2.c^2
    end
    Kref = X' * (B1 * (B2 * X))
    @test K ≈ Kref
end

# Testing directions: 

@testset "Identity tangent and normal" begin
    rng = MersenneTwister(12)
    n, p = 6, 3
    X = randn(rng, n, p)
    G = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :identity, :plain; k=n)

    sk = SketchLanding.IdentitySketch()
    N = zeros(n, p)
    T = zeros(n, p)

    SketchLanding.normal!(N, sk, sk, X, st)
    SketchLanding.tangent!(T, sk, sk, X, G, st)

    M = X' * X
    Nref = X * (M - I)
    K = X' * X
    P = G' * X
    Tref = G * K - X * P

    @test N ≈ Nref
    @test T ≈ Tref
end

@testset "Identity tangent works with solver step buffer" begin
    rng = MersenneTwister(21)
    n, p = 6, 3
    X = randn(rng, n, p)
    G = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :identity, :plain; k=n)
    sk = SketchLanding.IdentitySketch()

    SketchLanding.tangent!(st.Tstep, sk, sk, X, G, st)

    Tref = G * (X' * X) - X * (G' * X)
    @test st.Tstep ≈ Tref
end

@testset "Direction weights scale tangent and normal separately" begin
    rng = MersenneTwister(22)
    n, p = 6, 3
    X = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :identity, :plain; k=n)
    sk = SketchLanding.IdentitySketch()
    D = zeros(n, p)
    ωT = 2.5
    ωN = 0.25

    SketchLanding.direction!(D, sk, sk, SketchLanding.PlainEstimator(), prob, X, st; ωT=ωT, ωN=ωN, ξ=nothing, rng=rng)

    G = copy(X)
    Tref = tangent_reference(sk, sk, X, G)
    Nref = normal_reference(sk, sk, X)
    @test D ≈ ωT * Tref + ωN * Nref
end

@testset "State preallocates tangent and normal step buffers" begin
    n, p = 5, 2
    X = randn(n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :identity, :plain; k=n)

    @test size(st.Tstep) == size(X)
    @test size(st.Nstep) == size(X)
    @test st.Tstep == zeros(eltype(X), size(X)...)
    @test st.Nstep == zeros(eltype(X), size(X)...)
end

@testset "Custom function step-size uses iteration and state" begin
    n, p = 5, 2
    X = randn(n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :identity, :plain; k=n)
    st.D .= 3

    @test SketchLanding.compute_stepsize(Val(:function), (it, state) -> it + norm(state.D), 2, st) == 2 + norm(st.D)
end

@testset "Gaussian tangent matches dense reference" begin
    rng = MersenneTwister(456)
    n, p, k = 12, 4, 3
    X = randn(rng, n, p)
    G = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :gaussian, :plain; k=k)

    sk1 = SketchLanding.GaussianSketch(n, k)
    sk2 = SketchLanding.GaussianSketch(n, k)
    SketchLanding.sample!(sk1, rng)
    SketchLanding.sample!(sk2, rng)

    T = zeros(n, p)
    SketchLanding.tangent!(T, sk1, sk2, X, G, st)

    @test T ≈ tangent_reference(sk1, sk2, X, G) atol=1e-10 rtol=1e-10
end

@testset "Subsample tangent matches dense reference" begin
    rng = MersenneTwister(123)
    n, p, k = 10, 3, 4
    X = randn(rng, n, p)
    G = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :subsample, :plain; k=k)

    sk1 = SketchLanding.SubsampleSketch(n, k)
    sk2 = SketchLanding.SubsampleSketch(n, k)
    sk1.idx .= [1, 3, 7, 9]
    sk2.idx .= [2, 3, 8, 9]

    T = zeros(n, p)
    SketchLanding.tangent!(T, sk1, sk2, X, G, st)

    @test T ≈ tangent_reference(sk1, sk2, X, G) atol=1e-10 rtol=1e-10
end

@testset "Gaussian normal matches dense reference" begin
    rng = MersenneTwister(789)
    n, p, k = 12, 4, 3
    X = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :gaussian, :plain; k=k)

    sk1 = SketchLanding.GaussianSketch(n, k)
    sk2 = SketchLanding.GaussianSketch(n, k)
    SketchLanding.sample!(sk1, rng)
    SketchLanding.sample!(sk2, rng)

    N = zeros(n, p)
    SketchLanding.normal!(N, sk1, sk2, X, st)

    @test N ≈ normal_reference(sk1, sk2, X) atol=1e-10 rtol=1e-10
end

@testset "Haar sketch is on the Stiefel manifold" begin
    rng = MersenneTwister(2026)
    n, k = 12, 4
    sk = SketchLanding.HaarSketch(n, k)
    SketchLanding.sample!(sk, rng)

    # With default c = sqrt(n/k), Q = S/c is on the Stiefel manifold.
    Q = sk.S ./ sk.c
    @test Q' * Q ≈ I(k) atol=1e-10
    @test sk.S' * sk.S ≈ (sk.c^2) * I(k) atol=1e-10
end

@testset "Haar tangent and normal match dense reference" begin
    rng = MersenneTwister(31415)
    n, p, k = 12, 4, 3
    X = randn(rng, n, p)
    G = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :haar, :plain; k=k)

    sk1 = SketchLanding.HaarSketch(n, k)
    sk2 = SketchLanding.HaarSketch(n, k)
    SketchLanding.sample!(sk1, rng)
    SketchLanding.sample!(sk2, rng)

    T = zeros(n, p)
    SketchLanding.tangent!(T, sk1, sk2, X, G, st)
    @test T ≈ tangent_reference(sk1, sk2, X, G) atol=1e-10 rtol=1e-10

    N = zeros(n, p)
    SketchLanding.normal!(N, sk1, sk2, X, st)
    @test N ≈ normal_reference(sk1, sk2, X) atol=1e-10 rtol=1e-10
end

@testset "Subsample normal step matches dense reference and sparsity pattern" begin
    rng = MersenneTwister(321)
    n, p, k = 11, 4, 5
    X = randn(rng, n, p)
    prob = toy_problem(n, p)
    st = SketchLanding.init_state(prob, X, :subsample, :plain; k=k)

    sk1 = SketchLanding.SubsampleSketch(n, k)
    sk2 = SketchLanding.SubsampleSketch(n, k)
    sk1.idx .= [1, 3, 6, 8, 10]
    sk2.idx .= [2, 3, 7, 8, 11]

    N = randn(rng, n, p)
    SketchLanding.normal!(N, sk1, sk2, X, st)

    B1 = zeros(n, n)
    B2 = zeros(n, n)
    for i in sk1.idx
        B1[i, i] = sk1.c^2
    end
    for i in sk2.idx
        B2[i, i] = sk2.c^2
    end
    Nref = (B2 * X) * (X' * (B1 * X) - I)

    @test N ≈ Nref

    idx_set = Set(sk2.idx)
    for i in 1:n
        if !(i in idx_set)
            @test N[i, :] == zeros(p)
        end
    end
end

@testset "Normalized step-size scales by direction norm" begin
    rng = MersenneTwister(99)
    n, p = 6, 3
    X0 = randn(rng, n, p)
    prob = toy_problem(n, p)
    α0 = 2.0

    st = SketchLanding.init_state(prob, X0, :identity, :plain; k=n)
    sk = SketchLanding.IdentitySketch()
    Dref = zeros(n, p)
    SketchLanding.direction!(Dref, sk, sk, SketchLanding.PlainEstimator(), prob, X0, st;
        ωT=1.0,
        ωN=1.0,
        ξ=nothing,
        rng=rng,
    )
    αeff = α0 / max(1.0, norm(Dref))

    out = SketchLanding.sketchLanding(
        prob,
        X0;
        sketch=:identity,
        stepsize=α0,
        stepsize_mode=:normalized,
        maxiter=1,
        rng=MersenneTwister(99),
    )

    @test out.x_final ≈ X0 - αeff * Dref
end

@testset "Vector step-size uses per-iteration entries" begin
    rng = MersenneTwister(101)
    n, p = 6, 3
    X0 = randn(rng, n, p)
    prob = toy_problem(n, p)
    steps = [0.3]

    st = SketchLanding.init_state(prob, X0, :identity, :plain; k=n)
    sk = SketchLanding.IdentitySketch()
    Dref = zeros(n, p)
    SketchLanding.direction!(Dref, sk, sk, SketchLanding.PlainEstimator(), prob, X0, st;
        ωT=1.0,
        ωN=1.0,
        ξ=nothing,
        rng=rng,
    )

    out = SketchLanding.sketchLanding(
        prob,
        X0;
        sketch=:identity,
        stepsize=steps,
        stepsize_mode=:vector,
        maxiter=1,
        rng=MersenneTwister(101),
    )

    @test out.x_final ≈ X0 - steps[1] * Dref
end

@testset "Step-size mode validation is strict" begin
    n, p = 5, 2
    prob = toy_problem(n, p)
    X0 = randn(n, p)

    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; stepsize=it -> 1 / sqrt(it), stepsize_mode=:constant, maxiter=1)
    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; stepsize=1.0, stepsize_mode=:function, maxiter=1)
    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; stepsize=[0.1, 0.2], stepsize_mode=:vector, maxiter=3)
end

@testset "BB full-step matches hand-computed BB1" begin
    rng = MersenneTwister(2024)
    n, p = 6, 3
    X0 = randn(rng, n, p)
    prob = toy_problem(n, p)
    α0 = 0.05

    out = SketchLanding.sketchLanding(
        prob, X0;
        sketch=:identity,
        stepsize=α0,
        stepsize_mode=:bb,
        maxiter=4,
        rng=MersenneTwister(2024),
    )

    # Independently replay the BB1 recurrence using identity-sketch directions.
    st = SketchLanding.init_state(prob, X0, :identity, :plain; k=n)
    sk = SketchLanding.IdentitySketch()
    Xk = copy(X0)
    Xprev = nothing
    Dprev = nothing
    αlast = α0
    local Xref
    for it in 1:4
        D = zeros(n, p)
        SketchLanding.direction!(D, sk, sk, SketchLanding.PlainEstimator(), prob, Xk, st;
            ωT=1.0, ωN=1.0, ξ=nothing, rng=MersenneTwister(2024))
        if Xprev === nothing
            α = α0
        else
            s = Xk .- Xprev
            y = D .- Dprev
            den = dot(s, y)
            α = (isfinite(den) && abs(den) > 1e-12) ? dot(s, s) / den : αlast
        end
        Xprev = copy(Xk)
        Dprev = copy(D)
        αlast = α
        Xk = Xk .- α .* D
        Xref = Xk
    end

    @test out.x_final ≈ Xref atol=1e-10 rtol=1e-10
end

@testset "BB split applies separate α to tangent and normal" begin
    rng = MersenneTwister(99)
    n, p = 7, 2
    X0 = randn(rng, n, p)
    prob = toy_problem(n, p)
    α0 = 0.1
    ωT, ωN = 0.7, 1.3

    st = SketchLanding.init_state(prob, X0, :identity, :plain; k=n)
    sk = SketchLanding.IdentitySketch()

    Xk = copy(X0)
    Tprev = nothing
    Nprev = nothing
    αT_last = α0
    αN_last = α0
    bb1(s, y, fb) = begin
        den = dot(s, y)
        (isfinite(den) && abs(den) > 1e-12) ? dot(s, s) / den : fb
    end
    local Xref
    for it in 1:4
        D = zeros(n, p)
        SketchLanding.direction!(D, sk, sk, SketchLanding.PlainEstimator(), prob, Xk, st;
            ωT=ωT, ωN=ωN, ξ=nothing, rng=MersenneTwister(99))
        Tw = copy(st.Tstep)  # = ωT*T_k (directions.jl now weights both buffers)
        Nw = copy(st.Nstep)  # = ωN*N_k
        if Tprev === nothing
            αT = α0
            αN = α0
        else
            αT = bb1(-αT_last .* Tprev, Tw .- Tprev, αT_last)
            αN = bb1(-αN_last .* Nprev, Nw .- Nprev, αN_last)
        end
        Tprev = Tw
        Nprev = Nw
        αT_last = αT
        αN_last = αN
        Xk = Xk .- αT .* Tw .- αN .* Nw
        Xref = Xk
    end

    out = SketchLanding.sketchLanding(
        prob, X0;
        sketch=:identity,
        stepsize=α0,
        stepsize_mode=:bb_split,
        ωT=ωT,
        ωN=ωN,
        maxiter=4,
        rng=MersenneTwister(99),
    )

    @test out.x_final ≈ Xref atol=1e-10 rtol=1e-10
end

@testset "BB degenerate denominator reuses last α" begin
    n, p = 5, 2
    # Constant direction => s and y are collinear with s, but y = D_k - D_{k-1}.
    # With ωT=ωN=0 the direction is exactly zero, so ⟨s,y⟩ = 0 every step:
    # BB must fall back rather than produce NaN/Inf.
    prob = toy_problem(n, p)
    X0 = randn(MersenneTwister(5), n, p)

    out = SketchLanding.sketchLanding(
        prob, X0;
        sketch=:identity,
        stepsize=0.1,
        stepsize_mode=:bb,
        ωT=0.0,
        ωN=0.0,
        maxiter=5,
        rng=MersenneTwister(5),
    )

    @test all(isfinite, out.x_final)
    @test out.x_final ≈ X0 atol=1e-12  # zero direction => no movement, no blow-up

    out_split = SketchLanding.sketchLanding(
        prob, X0;
        sketch=:identity,
        stepsize=0.1,
        stepsize_mode=:bb_split,
        ωT=0.0,
        ωN=0.0,
        maxiter=5,
        rng=MersenneTwister(5),
    )
    @test all(isfinite, out_split.x_final)
    @test out_split.x_final ≈ X0 atol=1e-12
end

@testset "BB stepsize_mode validation" begin
    n, p = 5, 2
    prob = toy_problem(n, p)
    X0 = randn(n, p)

    # :bb / :bb_split are accepted with a scalar stepsize.
    @test SketchLanding.sketchLanding(prob, X0; sketch=:identity, stepsize=0.05, stepsize_mode=:bb, maxiter=2) isa SketchLanding.AlgOutput
    @test SketchLanding.sketchLanding(prob, X0; sketch=:identity, stepsize=0.05, stepsize_mode=:bb_split, maxiter=2) isa SketchLanding.AlgOutput

    # Non-scalar stepsize must be rejected for these modes.
    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; stepsize=[0.1, 0.2], stepsize_mode=:bb, maxiter=2)
    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; stepsize=(it -> 0.1), stepsize_mode=:bb_split, maxiter=2)
end

@testset "Recorder history levels" begin
    rec0 = SketchLanding.create_recorder(history=0)
    @test !rec0.record_cost
    @test !rec0.record_time
    @test !rec0.record_infeasibility
    @test !rec0.record_gradnorm
    @test !rec0.record_iterates
    @test !rec0.record_steps
    @test !rec0.record_norm_iterates

    rec1 = SketchLanding.create_recorder(history=1)
    @test rec1.record_cost
    @test rec1.record_time
    @test rec1.record_infeasibility
    @test rec1.record_gradnorm
    @test !rec1.record_iterates
    @test !rec1.record_steps
    @test rec1.record_norm_iterates

    rec2 = SketchLanding.create_recorder(history=2)
    @test rec2.record_cost
    @test rec2.record_time
    @test rec2.record_infeasibility
    @test rec2.record_gradnorm
    @test rec2.record_iterates
    @test rec2.record_steps
    @test rec2.record_norm_iterates
end

@testset "History 0 does not evaluate cost" begin
    n, p = 5, 2
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    bad_cost = X -> error("cost should not be called for history=0")
    prob = SketchLanding.FunctionProblem(n, p, gradf!; cost_fun=bad_cost)
    X0 = randn(n, p)

    out = SketchLanding.sketchLanding(
        prob,
        X0;
        maxiter=3,
        record=SketchLanding.create_recorder(history=0),
    )

    @test out.costs === nothing
    @test out.times === nothing
    @test out.infeasibilities === nothing
    @test out.gradnorms === nothing
    @test out.iterates === nothing
    @test out.step_tangent === nothing
    @test out.step_normal === nothing
    @test out.norm_iterates === nothing
end

@testset "Traces include iteration 0" begin
    n, p = 5, 2
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    costf = X -> sum(abs2, X)
    prob = SketchLanding.FunctionProblem(n, p, gradf!; cost_fun=costf)
    X0 = randn(n, p)
    maxiter = 4

    out = SketchLanding.sketchLanding(
        prob,
        X0;
        maxiter=maxiter,
        record=SketchLanding.create_recorder(history=1),
    )

    @test out.iterations == collect(0:maxiter)
    @test length(out.costs) == length(out.iterations)
    @test length(out.infeasibilities) == length(out.iterations)
    @test length(out.gradnorms) == length(out.iterations)
    @test length(out.times) == length(out.iterations)
    @test out.times[1] == 0.0
    @test out.costs[1] ≈ costf(X0)
end

@testset "Verbosity controls console output" begin
    n, p = 5, 2
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    prob = SketchLanding.FunctionProblem(n, p, gradf!)
    X0 = randn(n, p)

    function capture_stdout(f)
        return mktemp() do path, io
            redirect_stdout(io) do
                f()
                flush(stdout)
                flush(io)
            end
            seekstart(io)
            read(io, String)
        end
    end

    out0 = capture_stdout() do
        SketchLanding.sketchLanding(prob, X0; maxiter=3, verbosity=0)
    end
    @test isempty(out0)

    out1 = capture_stdout() do
        SketchLanding.sketchLanding(prob, X0; maxiter=3, verbosity=1)
    end
    # @test occursin("SketchLanding: starting run with maxiter=3", out1)
    # @test occursin("SketchLanding: finished with status=maxiter_reached after 3 iterations", out1)
    # @test !occursin("iter 1 gradnorm=", out1)

    out2 = capture_stdout() do
        SketchLanding.sketchLanding(prob, X0; maxiter=5, verbosity=2, print_every=2)
    end
    # @test occursin("SketchLanding: starting run with maxiter=5", out2)
    @test occursin("Iter 2: ||grad_T f(x)|| =", out2)
    @test occursin("Iter 4: ||grad_T f(x)|| =", out2)
    @test !occursin("Iter 1: ||grad_T f(x)|| =", out2)
    @test !occursin("Iter 5: ||grad_T f(x)|| =", out2)
    @test occursin("SketchLanding: finished with status=maxiter_reached after 5 iterations", out2)
end

@testset "Verbosity arguments are validated" begin
    n, p = 5, 2
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    prob = SketchLanding.FunctionProblem(n, p, gradf!)
    X0 = randn(n, p)

    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; verbosity=-1)
    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; verbosity=3)
    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; print_every=0)
end

@testset "Maxtime stops after a completed iteration" begin
    n, p = 5, 2
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    prob = SketchLanding.FunctionProblem(n, p, gradf!)
    X0 = randn(n, p)

    out = SketchLanding.sketchLanding(prob, X0; maxiter=10, maxtime=0.0, verbosity=0)

    @test out.iterations == collect(0:10)
    @test !isapprox(out.x_final, X0)
    @test out.times[2] >= 0.0
    @test out.times[3] == 0.0
end

@testset "Maxtime is reported in verbose output and validated" begin
    n, p = 5, 2
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    prob = SketchLanding.FunctionProblem(n, p, gradf!)
    X0 = randn(n, p)

    function capture_stdout(f)
        return mktemp() do path, io
            redirect_stdout(io) do
                f()
                flush(stdout)
                flush(io)
            end
            seekstart(io)
            read(io, String)
        end
    end

    out = capture_stdout() do
        SketchLanding.sketchLanding(prob, X0; maxiter=10, maxtime=0.0, verbosity=1)
    end

    @test occursin("status=maxtime_reached", out)
    @test occursin("after 1 iterations", out)
    @test occursin("maxtime=0.0s", out)

    @test_throws ArgumentError SketchLanding.sketchLanding(prob, X0; maxtime=-1.0)
end

@testset "Blow-up detection stops on non-finite iterates" begin
    n, p = 5, 2
    prob = toy_problem(n, p)
    X0 = randn(n, p)

    function capture_stdout(f)
        return mktemp() do path, io
            redirect_stdout(io) do
                f()
                flush(stdout)
                flush(io)
            end
            seekstart(io)
            read(io, String)
        end
    end

    out = capture_stdout() do
        SketchLanding.sketchLanding(
            prob,
            X0;
            stepsize=(it, state) -> Inf,
            stepsize_mode=:function,
            maxiter=10,
            verbosity=1,
        )
    end

    @test occursin("BLOW-UP: SketchLanding diverged!", out)
    @test occursin("status=blowup_detected", out)
    @test occursin("after 1 iterations", out)
end

@testset "Blow-up detection stops on large iterate norm" begin
    n, p = 5, 2
    prob = toy_problem(n, p)
    X0 = randn(n, p)

    function capture_stdout(f)
        return mktemp() do path, io
            redirect_stdout(io) do
                f()
                flush(stdout)
                flush(io)
            end
            seekstart(io)
            read(io, String)
        end
    end

    out = capture_stdout() do
        SketchLanding.sketchLanding(
            prob,
            X0;
            stepsize=1e8,
            maxiter=10,
            verbosity=1,
        )
    end

    @test occursin("BLOW-UP: SketchLanding diverged!", out)
    @test occursin("status=blowup_detected", out)
end
