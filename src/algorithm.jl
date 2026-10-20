function _build_sketch(sketch::Symbol, n::Int, k::Int, T::Type{<:Real})
    if sketch === :identity
        return IdentitySketch()
    elseif sketch === :gaussian
        return GaussianSketch(n, k; T=T)
    elseif sketch === :haar
        return HaarSketch(n, k; T=T)
    elseif sketch === :subsample
        return SubsampleSketch(n, k)
    else
        throw(ArgumentError("Unknown sketch kind: $sketch"))
    end
end

function _build_estimator(estimator::Symbol, n::Int, p::Int, Tperiod::Int, TT::Type{<:Real})
    if estimator === :plain
        return PlainEstimator()
    elseif estimator === :svrg
        return SVRGEstimator(Tperiod, n, p; T=TT)
    else
        throw(ArgumentError("Unknown estimator kind: $estimator"))
    end
end

function _current_infeasibility!(state::State)
    mul!(state.M, state.X', state.X) # M = X'*X
    copyto!(state.K, state.M)
    state.K .-= state.Ip
    return norm(state.K)
end

function _current_gradnorm!(prob::AbstractProblem, state::State)
    grad!(state.G, prob, state.X) # G = ∇f(X)
    mul!(state.M, state.X', state.X) # M = X'*X
    mul!(state.tmp_nxp, state.G, state.M) # tmp_nxp = G*M
    mul!(state.P, state.G', state.X) # P = G'*X
    mul!(state.BX1, state.X, state.P) # BX1 = X*P
    state.tmp_nxp .-= state.BX1
    return norm(state.tmp_nxp)
end

function _check_blowup(state::State, X0::AbstractMatrix)
    all(isfinite, state.X) || return true
    all(isfinite, state.D) || return true
    norm(state.X) <= 1e4 * max(1.0, norm(X0)) || return true
    return false
end

function _record_iteration!(out::AlgOutput, out_idx::Int, step_idx::Int, prob::AbstractProblem, state::State, record::RecordOptions, sk1, sk2, ωT, ωN, ξ)
    if record.record_cost
        c = cost(prob, state.X)
        out.costs[out_idx] = isnan(c) ? NaN : float(c)
    end

    if record.record_infeasibility
        out.infeasibilities[out_idx] = _current_infeasibility!(state)
    end

    if record.record_gradnorm
        out.gradnorms[out_idx] = _current_gradnorm!(prob, state)
    end

    if record.record_iterates
        out.iterates[out_idx] = Matrix{Float64}(state.X)
    end

    if record.record_steps && step_idx >= 1
        stoch_grad!(state.G, prob, state.X, ξ)
        tangent!(state.Tstep, sk1, sk2, state.X, state.G, state)
        normal!(state.Nstep, sk1, sk2, state.X, state)
        rmul!(state.Tstep, ωT) # Tstep = ωT*Tstep
        rmul!(state.Nstep, ωN) # Nstep = ωN*Nstep
        out.step_tangent[step_idx] = Matrix{Float64}(state.Tstep)
        out.step_normal[step_idx] = Matrix{Float64}(state.Nstep)
    end

    if record.record_norm_iterates
        out.norm_iterates[out_idx] = norm(state.X)
    end

    return nothing
end

"""
Run the Sketch Landing algorithm on `prob` starting from `X0` and return an `AlgOutput`.

Keyword arguments:
`sketch` selects the sketch family (`:subsample`, `:gaussian`, or `:identity`),
`k` is sketch size (ignored for `:identity`, where `k=n`),
`ωT` and `ωN` are the tangent and normal-direction weights,
`stepsize` is either a scalar base step-size for the built-in modes
(`:constant`, `:normalized`, `:decreasing`), a vector of per-iteration step-sizes
when `stepsize_mode=:vector`, or a custom function `(it, state) -> α_it`
when `stepsize_mode=:function`,
and `stepsize_mode` selects the step-size rule.
Estimator choice is controlled by `estimator` (`:plain` or `:svrg`),
with `T` the snapshot period for SVRG.
`maxiter` is the number of iterations,
`maxtime` is an optional wall-clock limit in seconds checked after each iteration,
`verbosity` controls console logging (`0`, `1`, or `2`),
`print_every` is the logging frequency used when `verbosity=2`,
`rng` is the random number generator,
`ξ_sampler` provides the stochastic sample `ξ` via `ξ_sampler(it)`,
`callback` is called as `(it, X, state)` after each iteration,
`record` controls optional logging in the output, 
and `snapshot_stochastic` controls whether SVRG uses stochastic or deterministic gradients at the snapshot point when forming `Dsketch(Y)`.
"""
function sketchLanding(prob::AbstractProblem, X0::AbstractMatrix;
    sketch::Symbol=:subsample,
    k::Int=1,
    ωT::Real=1.0,
    ωN::Real=1.0,
    stepsize=1e-2,
    stepsize_mode::Symbol=:constant,
    adagrad_ε::Real=1e-8,
    adam_β1::Real=0.9,
    adam_β2::Real=0.999,
    adam_ε::Real=1e-8,
    bb_tol::Real=1e-12,
    estimator::Symbol=:plain,
    T::Int=10,
    maxiter::Int=100,
    maxtime::Union{Nothing,Real}=nothing,
    verbosity::Int=0,
    print_every::Int=1,
    rng=Random.default_rng(),
    ξ_sampler=(it -> nothing),
    callback=(it, X, state) -> nothing,
    record::RecordOptions=RecordOptions(),
    snapshot_stochastic::Bool=false,
)
    nprob = n(prob)
    pprob = p(prob)
    size(X0, 1) == nprob || throw(ArgumentError("X0 row size must match n(prob)"))
    size(X0, 2) == pprob || throw(ArgumentError("X0 col size must match p(prob)"))
    verbosity in 0:2 || throw(ArgumentError("verbosity must be 0, 1, or 2"))
    print_every >= 1 || throw(ArgumentError("print_every must be >= 1"))
    isnothing(maxtime) || maxtime >= 0 || throw(ArgumentError("maxtime must be >= 0"))
    stepsize_mode in (:constant, :normalized, :decreasing, :vector, :function, :adagrad, :adam, :bb, :bb_split) || throw(ArgumentError("stepsize_mode must be one of :constant, :normalized, :decreasing, :vector, :function, :adagrad, :adam, :bb, :bb_split"))
    if stepsize_mode === :function
        stepsize isa Function || throw(ArgumentError("stepsize must be a function `(it, state) -> α_it` when stepsize_mode=:function"))
    elseif stepsize_mode === :vector
        stepsize isa AbstractVector || throw(ArgumentError("stepsize must be a vector when stepsize_mode=:vector"))
        maxiter <= length(stepsize) || throw(ArgumentError("maxiter must be <= length(stepsize) when stepsize_mode=:vector"))
    else
        stepsize isa Real || throw(ArgumentError("stepsize must be a scalar when stepsize_mode=$stepsize_mode"))
    end

    kk = (k == 0) ? max(1, min(nprob, pprob)) : k
    if sketch === :identity
        kk = nprob
    end
    kk <= nprob || throw(ArgumentError("k must satisfy k <= n(prob)"))

    TT = eltype(X0)
    state = init_state(prob, X0, sketch, estimator; k=kk)

    sk1 = _build_sketch(sketch, nprob, kk, TT)
    sk2 = _build_sketch(sketch, nprob, kk, TT)
    est = _build_estimator(estimator, nprob, pprob, T, TT)
    out = init_output(X0, maxiter, record)
    ss = _build_stepsize_state(stepsize_mode, stepsize, nprob, pprob, TT; adagrad_ε=adagrad_ε, adam_β1=adam_β1, adam_β2=adam_β2, adam_ε=adam_ε, bb_tol=bb_tol)

    accumulated_compute = 0.0
    run_start = time_ns()
    completed_iters = 0
    stop_reason = :maxiter_reached
    if record.record_time
        out.times[1] = 0.0
    end
    _record_iteration!(out, 1, 0, prob, state, record, sk1, sk2, ωT, ωN, nothing)

    if verbosity >= 1
        @printf("Running SketchLanding with maxiter=%d, sketch=%s, k=%d, estimator=%s...\n", maxiter, sketch, kk, estimator)
    end

    for it in 1:maxiter
        tstart = time_ns()
        sample!(sk1, rng)
        sample!(sk2, rng)

        update_estimator!(est, prob, state.X, it, state, rng; ωT=ωT, ωN=ωN) # does nothing for plain estimator
        ξ = ξ_sampler(it)

        if est isa SVRGEstimator
            direction!(state.D, sk1, sk2, est, prob, state.X, state;
                ωT=ωT,
                ωN=ωN,
                ξ=ξ,
                rng=rng,
                snapshot_stochastic=snapshot_stochastic,
            )
        else
            direction!(state.D, sk1, sk2, est, prob, state.X, state; ωT=ωT, ωN=ωN, ξ=ξ, rng=rng)
        end

        αit = apply_step!(state.X, state.D, ss, it, state)

        tend = time_ns()
        accumulated_compute += (tend - tstart) / 1e9
        if record.record_time
            out.times[it + 1] = accumulated_compute
        end

        _record_iteration!(out, it + 1, it, prob, state, record, sk1, sk2, ωT, ωN, ξ)
        if verbosity == 2 && mod(it, print_every) == 0
            gradnorm = _current_gradnorm!(prob, state)
            infeasibility = _current_infeasibility!(state)
            @printf("Iter %d: ||grad_T f(x)|| = %e, ||c(x)|| = %e, stepsize = %e\n", it, gradnorm, infeasibility, αit)
        end
        callback(it, state.X, state)
        completed_iters = it

        elapsed_wall = (time_ns() - run_start) / 1e9
        if !isnothing(maxtime) && elapsed_wall >= maxtime
            stop_reason = :maxtime_reached
            break
        end
        if _check_blowup(state, X0)
            stop_reason = :blowup_detected
            println("BLOW-UP: SketchLanding diverged!")
            break
        end
    end

    out.x_final .= Matrix{Float64}(state.X)
    if verbosity >= 1
        if stop_reason === :maxtime_reached
            println("SketchLanding: finished with status=maxtime_reached after $completed_iters iterations (maxtime=$(Float64(maxtime))s)")
        elseif stop_reason === :blowup_detected
            println("SketchLanding: finished with status=blowup_detected after $completed_iters iterations")
        else
            println("SketchLanding: finished with status=maxiter_reached after $completed_iters iterations")
        end
    end
    return out
end
