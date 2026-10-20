"""Base type for all step-size states."""
abstract type AbstractStepsizeState end

"""
Wraps the existing scalar step-size modes (:constant, :normalized, :decreasing,
:vector, :function) so they can be used through the unified `apply_step!` interface.
"""
struct ScalarStepsizeState <: AbstractStepsizeState
    mode::Symbol
    stepsize  # Real, AbstractVector, or Function
end

"""
Adagrad step-size state.

Accumulates the element-wise sum of squared directions in `v` and applies
    X -= α * D / (√v + ε)
at each iteration. `α` is the base learning rate and `ε` prevents division by zero.
"""
mutable struct AdagradStepsizeState{T<:Real} <: AbstractStepsizeState
    α::T
    ε::T
    v::Matrix{T}
end

function AdagradStepsizeState(α::Real, n::Int, p::Int; ε::Real=1e-8, T::Type{<:Real}=Float64)
    TT = float(T)
    return AdagradStepsizeState{TT}(TT(α), TT(ε), zeros(TT, n, p))
end

"""
Adam step-size state.

Maintains first-moment estimate `m` and second-moment estimate `v`, with bias
correction, and applies
    m  = β1*m + (1-β1)*D
    v  = β2*v + (1-β2)*D⊙D
    m̂  = m / (1 - β1^t)
    v̂  = v / (1 - β2^t)
    X -= α * m̂ / (√v̂ + ε)
Standard defaults: β1=0.9, β2=0.999, ε=1e-8.
"""
mutable struct AdamStepsizeState{T<:Real} <: AbstractStepsizeState
    α::T
    β1::T
    β2::T
    ε::T
    m::Matrix{T}   # first moment
    v::Matrix{T}   # second moment
end

function AdamStepsizeState(α::Real, n::Int, p::Int;
    β1::Real=0.9, β2::Real=0.999, ε::Real=1e-8, T::Type{<:Real}=Float64,
)
    TT = float(T)
    return AdamStepsizeState{TT}(TT(α), TT(β1), TT(β2), TT(ε), zeros(TT, n, p), zeros(TT, n, p))
end

"""
Barzilai–Borwein (BB1, long-step) state using the full landing direction.

With `s = X_k − X_{k−1}` and `y = D_k − D_{k−1}`, the step size is
    α = ⟨s, s⟩ / ⟨s, y⟩
using the Frobenius inner product. Iteration 1 uses `α₀`. If `⟨s, y⟩` is
non-finite or `|⟨s, y⟩| ≤ tol`, the previously accepted `α` is reused
(`α₀` if none accepted yet).
"""
mutable struct BBStepsizeState{T<:Real} <: AbstractStepsizeState
    α₀::T
    tol::T
    α_last::T
    has_prev::Bool
    X_prev::Matrix{T}
    D_prev::Matrix{T}
end

function BBStepsizeState(α::Real, n::Int, p::Int; tol::Real=1e-12, T::Type{<:Real}=Float64)
    TT = float(T)
    return BBStepsizeState{TT}(TT(α), TT(tol), TT(α), false, zeros(TT, n, p), zeros(TT, n, p))
end

"""
Barzilai–Borwein (BB1, long-step) state applied separately to the tangent and
normal components.

`Tstep = ωT·T` and `Nstep = ωN·N` are read from the algorithm `state`. Per
component `c` with previous weighted component `c_{k−1}` and previously
accepted `α_c`, the actual previous displacement is `s_c = −α_c · c_{k−1}` and
`y_c = c_k − c_{k−1}`, giving `α_c = ⟨s_c, s_c⟩ / ⟨s_c, y_c⟩`. The update is
`X .-= α_T·(ωT·T) .+ α_N·(ωN·N)`. Same iteration-1 / degenerate-denominator
fallback as `BBStepsizeState`.
"""
mutable struct BBSplitStepsizeState{T<:Real} <: AbstractStepsizeState
    α₀::T
    tol::T
    αT_last::T
    αN_last::T
    has_prev::Bool
    T_prev::Matrix{T}
    N_prev::Matrix{T}
end

function BBSplitStepsizeState(α::Real, n::Int, p::Int; tol::Real=1e-12, T::Type{<:Real}=Float64)
    TT = float(T)
    return BBSplitStepsizeState{TT}(TT(α), TT(tol), TT(α), TT(α), false, zeros(TT, n, p), zeros(TT, n, p))
end

"""
Build the step-size state from the algorithm keyword arguments.

For scalar modes, wraps the existing `compute_stepsize` logic.
For `:adagrad`, initialises an `AdagradStepsizeState` with a zero accumulator.
For `:adam`, initialises an `AdamStepsizeState` with zero moment estimates.
For `:bb`/`:bb_split`, initialises the corresponding BB state with empty history.
"""
function _build_stepsize_state(mode::Symbol, stepsize, n::Int, p::Int, T::Type{<:Real};
    adagrad_ε::Real=1e-8,
    adam_β1::Real=0.9,
    adam_β2::Real=0.999,
    adam_ε::Real=1e-8,
    bb_tol::Real=1e-12,
)
    if mode === :adagrad
        return AdagradStepsizeState(float(stepsize), n, p; ε=adagrad_ε, T=T)
    elseif mode === :adam
        return AdamStepsizeState(float(stepsize), n, p; β1=adam_β1, β2=adam_β2, ε=adam_ε, T=T)
    elseif mode === :bb
        return BBStepsizeState(float(stepsize), n, p; tol=bb_tol, T=T)
    elseif mode === :bb_split
        return BBSplitStepsizeState(float(stepsize), n, p; tol=bb_tol, T=T)
    else
        return ScalarStepsizeState(mode, stepsize)
    end
end

"""BB1 ratio ⟨s,s⟩/⟨s,y⟩ with finite/tol guard; returns `α_fallback` if degenerate."""
function _bb1(s::AbstractArray, y::AbstractArray, tol::Real, α_fallback::Real)
    den = dot(s, y)
    (isfinite(den) && abs(den) > tol) || return α_fallback
    α = dot(s, s) / den
    return isfinite(α) ? α : α_fallback
end

"""
Apply one descent step `X -= step(D)` and return a representative scalar step-size
for logging (base `α` for element-wise modes).
"""
function apply_step!(X::AbstractMatrix, D::AbstractMatrix, ss::ScalarStepsizeState, it::Int, state)
    α = if ss.mode === :function
        compute_stepsize(Val(:function), ss.stepsize, it, state)
    elseif ss.mode === :vector
        compute_stepsize(Val(:vector), ss.stepsize, it, state)
    else
        compute_stepsize(ss.mode, ss.stepsize, it, state)
    end
    axpy!(-α, D, X)
    return float(α)
end

function apply_step!(X::AbstractMatrix, D::AbstractMatrix, ss::AdagradStepsizeState, it::Int, state)
    ss.v .+= D .* D
    X .-= ss.α .* D ./ (sqrt.(ss.v) .+ ss.ε)
    return ss.α
end

function apply_step!(X::AbstractMatrix, D::AbstractMatrix, ss::AdamStepsizeState, it::Int, state)
    ss.m .= ss.β1 .* ss.m .+ (1 - ss.β1) .* D
    ss.v .= ss.β2 .* ss.v .+ (1 - ss.β2) .* D .* D
    bias1 = 1 - ss.β1^it
    bias2 = 1 - ss.β2^it
    X .-= ss.α .* (ss.m ./ bias1) ./ (sqrt.(ss.v ./ bias2) .+ ss.ε)
    return ss.α
end

function apply_step!(X::AbstractMatrix, D::AbstractMatrix, ss::BBStepsizeState, it::Int, state)
    if !ss.has_prev
        α = ss.α₀
    else
        s = X .- ss.X_prev          # s = X_k − X_{k−1}
        y = D .- ss.D_prev          # y = D_k − D_{k−1}
        α = _bb1(s, y, ss.tol, ss.α_last)
    end
    # Snapshot X_k, D_k as the new "previous" BEFORE mutating X.
    copyto!(ss.X_prev, X)
    copyto!(ss.D_prev, D)
    ss.has_prev = true
    ss.α_last = α
    axpy!(-α, D, X)                 # X -= α*D
    return float(α)
end

function apply_step!(X::AbstractMatrix, D::AbstractMatrix, ss::BBSplitStepsizeState, it::Int, state)
    Tw = state.Tstep                # = ωT*T_k
    Nw = state.Nstep                # = ωN*N_k
    if !ss.has_prev
        αT = ss.α₀
        αN = ss.α₀
    else
        sT = (-ss.αT_last) .* ss.T_prev   # s_T = −α_T · (ωT·T_{k−1})
        yT = Tw .- ss.T_prev              # y_T = ωT·T_k − ωT·T_{k−1}
        αT = _bb1(sT, yT, ss.tol, ss.αT_last)
        sN = (-ss.αN_last) .* ss.N_prev
        yN = Nw .- ss.N_prev
        αN = _bb1(sN, yN, ss.tol, ss.αN_last)
    end
    copyto!(ss.T_prev, Tw)
    copyto!(ss.N_prev, Nw)
    ss.has_prev = true
    ss.αT_last = αT
    ss.αN_last = αN
    X .-= αT .* Tw .+ αN .* Nw      # fused per-component descent; D ignored
    return float((αT + αN) / 2)
end
