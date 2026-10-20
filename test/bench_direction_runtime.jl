using Random
using LinearAlgebra
using DelimitedFiles
using BenchmarkTools
using Statistics
using SketchLanding

const HAS_PLOTS = Base.find_package("Plots") !== nothing
if HAS_PLOTS
    @eval using Plots
end

function toy_problem(nv, pv)
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    return SketchLanding.FunctionProblem(nv, pv, gradf!)
end

function bench_runtime_direction!(sk1, sk2, prob, X, st; ωT::Real=1.0, ωN::Real=1.0, show_btime::Bool=false)
    D = zeros(eltype(X), size(X, 1), size(X, 2))
    est = SketchLanding.PlainEstimator()
    rng = MersenneTwister(0)
    ξ = nothing

    SketchLanding.direction!(D, sk1, sk2, est, prob, X, st; ωT=ωT, ωN=ωN, ξ=ξ, rng=rng)

    if show_btime
        @btime SketchLanding.direction!($D, $sk1, $sk2, $est, $prob, $X, $st; ωT=$ωT, ωN=$ωN, ξ=$ξ, rng=$rng)
    end

    return @belapsed SketchLanding.direction!($D, $sk1, $sk2, $est, $prob, $X, $st; ωT=$ωT, ωN=$ωN, ξ=$ξ, rng=$rng)
end

function empirical_slope(ns, t)
    x = log.(Float64.(ns))
    y = log.(t)
    return cov(x, y) / var(x)
end

function benchmark_direction_runtime(; n_values=100:100:1000, seed::Int=2026, ωT::Real=1.0, ωN::Real=1.0)
    default_threads = BLAS.get_num_threads()
    thread_modes = default_threads == 1 ? [1] : [1, default_threads]

    ns = collect(n_values)
    t_identity = Dict{Int, Vector{Float64}}()
    t_gaussian = Dict{Int, Vector{Float64}}()
    t_subsample = Dict{Int, Vector{Float64}}()

    println("Default BLAS threads: $default_threads")
    println("Benchmarking thread modes: $(join(thread_modes, ", "))")

    for nt in thread_modes
        BLAS.set_num_threads(nt)
        println("\n--- BLAS threads = $nt ---")

        rng = MersenneTwister(seed)
        tI = zeros(Float64, length(ns))
        tG = zeros(Float64, length(ns))
        tS = zeros(Float64, length(ns))

        for (j, n) in enumerate(ns)
            p = n
            k = max(1, n ÷ 10)
            prob = toy_problem(n, p)
            X = randn(rng, n, p)

            skI = SketchLanding.IdentitySketch()
            stI = SketchLanding.init_state(prob, X, :identity, :plain; k=n)

            skG = SketchLanding.GaussianSketch(n, k)
            SketchLanding.sample!(skG, rng)
            stG = SketchLanding.init_state(prob, X, :gaussian, :plain; k=k)

            skS = SketchLanding.SubsampleSketch(n, k)
            SketchLanding.sample!(skS, rng)
            stS = SketchLanding.init_state(prob, X, :subsample, :plain; k=k)

            println("n=$n, p=$p, k=$k | Identity plain direction!")
            tI[j] = bench_runtime_direction!(skI, skI, prob, X, stI; ωT=ωT, ωN=ωN, show_btime=false)

            println("n=$n, p=$p, k=$k | Gaussian plain direction!")
            tG[j] = bench_runtime_direction!(skG, skG, prob, X, stG; ωT=ωT, ωN=ωN, show_btime=false)

            println("n=$n, p=$p, k=$k | Subsample plain direction!")
            tS[j] = bench_runtime_direction!(skS, skS, prob, X, stS; ωT=ωT, ωN=ωN, show_btime=false)
        end

        t_identity[nt] = tI
        t_gaussian[nt] = tG
        t_subsample[nt] = tS
    end

    BLAS.set_num_threads(default_threads)
    ref_thread = thread_modes[end]

    println("Identity slope (BLAS=$ref_thread) ≈ ", empirical_slope(ns, t_identity[ref_thread]))
    println("Gaussian slope (BLAS=$ref_thread) ≈ ", empirical_slope(ns, t_gaussian[ref_thread]))
    println("Subsample slope (BLAS=$ref_thread) ≈ ", empirical_slope(ns, t_subsample[ref_thread]))

    out_csv = joinpath(@__DIR__, "direction_runtime_data.csv")
    if length(thread_modes) == 1
        nt = thread_modes[1]
        data = hcat(Float64.(ns), t_identity[nt], t_gaussian[nt], t_subsample[nt])
    else
        nt1, ntd = thread_modes[1], thread_modes[2]
        data = hcat(
            Float64.(ns),
            t_identity[nt1],
            t_gaussian[nt1],
            t_subsample[nt1],
            t_identity[ntd],
            t_gaussian[ntd],
            t_subsample[ntd],
        )
    end
    writedlm(out_csv, data, ',')
    println("\nSaved runtime data to: $out_csv")

    if HAS_PLOTS
        plt = Plots.plot(
            ;
            xlabel="n (with p=n, k=n/10)",
            ylabel="runtime of direction! (seconds)",
            title="Runtime of plain sketch direction vs n",
        )

        for nt in thread_modes
            Plots.plot!(plt, ns, t_identity[nt]; label="Identity (BLAS=$nt)", lw=2, marker=:circle)
            Plots.plot!(plt, ns, t_gaussian[nt]; label="Gaussian (BLAS=$nt)", lw=2, marker=:square)
            Plots.plot!(plt, ns, t_subsample[nt]; label="Subsample (BLAS=$nt)", lw=2, marker=:diamond)
        end

        out_png = joinpath(@__DIR__, "direction_runtime_plot.png")
        Plots.savefig(plt, out_png)
        println("Saved plot to: $out_png")

        plt_loglog = Plots.plot(
            ;
            xscale=:log10,
            yscale=:log10,
            xlabel="n (with p=n, k=n/10)",
            ylabel="runtime of direction! (seconds)",
            title="Runtime of plain sketch direction vs n (log-log)",
        )

        for nt in thread_modes
            Plots.plot!(plt_loglog, ns, t_identity[nt]; label="Identity (BLAS=$nt)", lw=2, marker=:circle)
            Plots.plot!(plt_loglog, ns, t_gaussian[nt]; label="Gaussian (BLAS=$nt)", lw=2, marker=:square)
            Plots.plot!(plt_loglog, ns, t_subsample[nt]; label="Subsample (BLAS=$nt)", lw=2, marker=:diamond)
        end

        n0 = Float64(ns[1])
        t0 = min(t_identity[ref_thread][1], t_gaussian[ref_thread][1], t_subsample[ref_thread][1])
        slope2 = t0 .* (Float64.(ns) ./ n0) .^ 2
        slope3 = t0 .* (Float64.(ns) ./ n0) .^ 3
        Plots.plot!(plt_loglog, ns, slope2; label="slope 2", lw=2, ls=:dash, color=:black)
        Plots.plot!(plt_loglog, ns, slope3; label="slope 3", lw=2, ls=:dot, color=:gray40)

        out_loglog_png = joinpath(@__DIR__, "direction_runtime_plot_loglog.png")
        Plots.savefig(plt_loglog, out_loglog_png)
        println("Saved log-log plot to: $out_loglog_png")
    else
        println("Plots.jl not found in this environment.")
    end

    return ns, t_identity, t_gaussian, t_subsample
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_direction_runtime()
end
