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

function bench_direct_gram!(M, X; show_btime::Bool=false)
    mul!(M, X', X) # warm-up
    if show_btime
        @btime mul!($M, $(X'), $X)
    end
    return @belapsed mul!($M, $(X'), $X)
end

function bench_sketched_gram!(M, SX, S, X; show_btime::Bool=false)
    mul!(SX, S', X) # warm-up
    mul!(M, SX', SX) # warm-up

    if show_btime
        @btime begin
            mul!($SX, $(S'), $X)
            mul!($M, $(SX'), $SX)
        end
    end

    return @belapsed begin
        mul!($SX, $(S'), $X)
        mul!($M, $(SX'), $SX)
    end
end

function toy_problem(nv, pv)
    gradf! = (G, X) -> (copyto!(G, X); nothing)
    return SketchLanding.FunctionProblem(nv, pv, gradf!)
end

function bench_xtbx_subsample!(M, sk_sub, X, st; show_btime::Bool=false)
    SketchLanding.XtBX!(M, sk_sub, X, st) # warm-up
    if show_btime
        @btime SketchLanding.XtBX!($M, $sk_sub, $X, $st)
    end
    return @belapsed SketchLanding.XtBX!($M, $sk_sub, $X, $st)
end

function empirical_slope(ns, t)
    x = log.(Float64.(ns))
    y = log.(t)
    return cov(x, y) / var(x)
end

function benchmark_xtx_vs_sketched_gram(; n_values=100:100:2000, seed::Int=2026)
    default_threads = BLAS.get_num_threads()
    thread_modes = default_threads == 1 ? [1] : [1, default_threads]

    ns = collect(n_values)
    t_direct = Dict{Int, Vector{Float64}}()
    t_sketched = Dict{Int, Vector{Float64}}()
    t_xtbx_sub = Dict{Int, Vector{Float64}}()

    println("Default BLAS threads: $default_threads")
    println("Benchmarking thread modes: $(join(thread_modes, ", "))")

    for nt in thread_modes
        BLAS.set_num_threads(nt)
        println("\n--- BLAS threads = $nt ---")

        rng = MersenneTwister(seed)
        td = zeros(Float64, length(ns))
        ts = zeros(Float64, length(ns))
        tx = zeros(Float64, length(ns))
        for (j, n) in enumerate(ns)
            p = n
            k = max(1, n ÷ 10)

            X = randn(rng, n, p)
            S = randn(rng, n, k) / sqrt(k)
            M = zeros(Float64, p, p)
            SX = zeros(Float64, k, p)
            prob = toy_problem(n, p)
            st = SketchLanding.init_state(prob, X, :subsample, :plain; k=k)
            sk_sub = SketchLanding.SubsampleSketch(n, k)
            SketchLanding.sample!(sk_sub, rng)

            println("n=$n, p=$p, k=$k | direct mul!(M, X', X)")
            td[j] = bench_direct_gram!(M, X; show_btime=true)

            println("n=$n, p=$p, k=$k | sketched mul!(SX, S', X); mul!(M, SX', SX)")
            ts[j] = bench_sketched_gram!(M, SX, S, X; show_btime=true)

            println("n=$n, p=$p, k=$k | XtBX!(M, sk_sub, X, state)")
            tx[j] = bench_xtbx_subsample!(M, sk_sub, X, st; show_btime=true)
        end
        t_direct[nt] = td
        t_sketched[nt] = ts
        t_xtbx_sub[nt] = tx
    end

    BLAS.set_num_threads(default_threads)
    ref_thread = thread_modes[end]

    println("Direct slope (BLAS=$ref_thread) ≈ ", empirical_slope(ns, t_direct[ref_thread]))
    println("Sketched slope (BLAS=$ref_thread) ≈ ", empirical_slope(ns, t_sketched[ref_thread]))
    println("XtBX Subsample slope (BLAS=$ref_thread) ≈ ", empirical_slope(ns, t_xtbx_sub[ref_thread]))

    out_csv = joinpath(@__DIR__, "xtx_vs_sketched_gram_data.csv")
    if length(thread_modes) == 1
        nt = thread_modes[1]
        ratio = t_sketched[nt] ./ t_direct[nt]
        ratio_xtbx = t_xtbx_sub[nt] ./ t_direct[nt]
        data = hcat(Float64.(ns), t_direct[nt], t_sketched[nt], t_xtbx_sub[nt], ratio, ratio_xtbx)
    else
        nt1, ntd = thread_modes[1], thread_modes[2]
        ratio1 = t_sketched[nt1] ./ t_direct[nt1]
        ratiod = t_sketched[ntd] ./ t_direct[ntd]
        ratiox1 = t_xtbx_sub[nt1] ./ t_direct[nt1]
        ratioxd = t_xtbx_sub[ntd] ./ t_direct[ntd]
        data = hcat(
            Float64.(ns),
            t_direct[nt1],
            t_sketched[nt1],
            t_xtbx_sub[nt1],
            ratio1,
            ratiox1,
            t_direct[ntd],
            t_sketched[ntd],
            t_xtbx_sub[ntd],
            ratiod,
            ratioxd,
        )
    end
    writedlm(out_csv, data, ',')
    println("\nSaved runtime data to: $out_csv")

    if HAS_PLOTS
        plt = Plots.plot(; xlabel="n (with p=n, k=n/10)", ylabel="runtime (seconds)", title="Direct Gram vs Sketched Gram")
        for nt in thread_modes
            Plots.plot!(plt, ns, t_direct[nt]; label="mul!(M, X', X) (BLAS=$nt)", lw=2, marker=:circle)
            Plots.plot!(plt, ns, t_sketched[nt]; label="sketched (BLAS=$nt)", lw=2, marker=:square)
            Plots.plot!(plt, ns, t_xtbx_sub[nt]; label="XtBX subsample (BLAS=$nt)", lw=2, marker=:diamond)
        end
        out_png = joinpath(@__DIR__, "xtx_vs_sketched_gram_plot.png")
        Plots.savefig(plt, out_png)
        println("Saved plot to: $out_png")

        plt_loglog = Plots.plot(
            ;
            xscale=:log10,
            yscale=:log10,
            xlabel="n (with p=n, k=n/10)",
            ylabel="runtime (seconds)",
            title="Direct Gram vs Sketched Gram (log-log)",
        )
        for nt in thread_modes
            Plots.plot!(plt_loglog, ns, t_direct[nt]; label="mul!(M, X', X) (BLAS=$nt)", lw=2, marker=:circle)
            Plots.plot!(plt_loglog, ns, t_sketched[nt]; label="sketched (BLAS=$nt)", lw=2, marker=:square)
            Plots.plot!(plt_loglog, ns, t_xtbx_sub[nt]; label="XtBX subsample (BLAS=$nt)", lw=2, marker=:diamond)
        end

        n0 = Float64(ns[1])
        t0 = min(t_direct[ref_thread][1], t_sketched[ref_thread][1], t_xtbx_sub[ref_thread][1])
        slope2 = t0 .* (Float64.(ns) ./ n0) .^ 2
        slope3 = t0 .* (Float64.(ns) ./ n0) .^ 3
        Plots.plot!(plt_loglog, ns, slope2; label="slope 2", lw=2, ls=:dash, color=:black)
        Plots.plot!(plt_loglog, ns, slope3; label="slope 3", lw=2, ls=:dot, color=:gray40)

        out_loglog_png = joinpath(@__DIR__, "xtx_vs_sketched_gram_plot_loglog.png")
        Plots.savefig(plt_loglog, out_loglog_png)
        println("Saved log-log plot to: $out_loglog_png")
    else
        println("Plots.jl not found in this environment.")
    end

    return ns, t_direct, t_sketched, t_xtbx_sub
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_xtx_vs_sketched_gram()
end
