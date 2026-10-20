"""Base type for all sketch operators."""
abstract type AbstractSketch end

"""Base type for variance estimators."""
abstract type AbstractEstimator end

"""Identity sketch, i.e. B = I."""
struct IdentitySketch <: AbstractSketch end

"""
Sketches storing a dense `n×k` matrix `S` with `B = S*S'`.

Subtypes share the same algebraic action on `S`; they only differ in how `sample!`
draws `S`.
"""
abstract type DenseMatrixSketch <: AbstractSketch end

"""
Gaussian sketch storing S with B = S*S'.

`c` scales i.i.d. standard normal entries in `S` during sampling.
"""
struct GaussianSketch{T} <: DenseMatrixSketch
    n::Int
    k::Int
    c::T
    S::Matrix{T}
end

"""
Haar sketch storing S with B = S*S'.

`S` is drawn uniformly from the Stiefel manifold (orthonormal columns) via the
QR decomposition of an `n×k` standard Gaussian matrix, then scaled by `c`. With
the default `c = sqrt(n/k)`, `E[S*S'] = I_n`.
"""
struct HaarSketch{T} <: DenseMatrixSketch
    n::Int
    k::Int
    c::T
    S::Matrix{T}
    G::Matrix{T} # scratch n×k Gaussian buffer for sampling
end

"""
Subsampling sketch storing selected indices only, with B diagonal on sampled rows.

For sampled rows `idx`, B acts as multiplication by `c^2`.
"""
struct SubsampleSketch <: AbstractSketch
    n::Int
    k::Int
    c::Float64
    idx::Vector{Int}
end

"""Plain estimator: use current sketched direction only."""
struct PlainEstimator <: AbstractEstimator end

"""
SVRG estimator storing snapshot point and cached directions.

Snapshot update period is `T` iterations.
"""
struct SVRGEstimator{T} <: AbstractEstimator
    T::Int
    snapshotX::Matrix{T}
    snapshotDfull::Matrix{T}
    snapshotDsketch::Matrix{T}
end

"""
Workspace state holding iterate, gradients, and preallocated scratch buffers.

For sketches `B1, B2` and iterate `X`, the algorithm uses:

- `N = (B2*X) * (X'*(B1*X) - I)` (normal component)
- `T = G*K - (B1*X)*P` (tangent component)
  with
  `K = X'*(B1*(B2*X))` and `P = G'*(B2*X)`.

Main buffers:
- `X`: current iterate.
- `G`: (stochastic or deterministic) gradient at `X`.
- `D`: search direction (typically `D = ωT*T + ωN*N`).
- `Tstep`: preallocated `n×p` buffer storing the tangent component of the direction.
- `Nstep`: preallocated `n×p` buffer storing the normal component of the direction.
- `BX1`: stores `B1*X` (or another `B*X` intermediate).
- `BX2`: stores `B2*X` (or another `B*X`/product intermediate).
- `tmp_nxp`: temporary `n×p` workspace (e.g. `B1*(B2*X)` or `G*(X'X)`).
- `M`: `p×p` Gram-like matrix, usually `X'*(B1*X)` or `X'X`.
- `K`: `p×p` matrix `X'*(B1*(B2*X))` used in tangent terms.
- `P`: `p×p` matrix `G'*(B2*X)` (or similar `G'X` temporary).
- `Ip`: identity matrix `I_p`.

Sketch-specific temporaries:
- `SX1`, `SX2`: `k×p` workspaces for Gaussian sketch products (`S'X`, etc.).
- `temp_kxp`: temporary `k×p` workspace for sketch-space products.
- `temp_kxk`: temporary `k×k` workspace for Gaussian sketch Gram-like products.
- `temp_nxk`, `temp_nxk2`: temporary `n×k` workspaces for Gaussian tangent products.
- `X_I`, `X_J`, `X_K`: gathered sampled rows used by subsampling routines.
- `G_J`: gathered gradient rows for subsampling-specific computations.
- `inter_posI`, `inter_posJ`, `r`: intersection bookkeeping for sampled index sets.
"""
mutable struct State{T}
    X::Matrix{T}
    G::Matrix{T}
    D::Matrix{T}
    Tstep::Matrix{T}
    Nstep::Matrix{T}
    BX1::Matrix{T}
    BX2::Matrix{T}
    tmp_nxp::Matrix{T}
    M::Matrix{T}
    K::Matrix{T}
    P::Matrix{T}
    Ip::Matrix{T}
    SX1::Matrix{T}
    SX2::Matrix{T}
    temp_kxp::Matrix{T}
    temp_kxk::Matrix{T}
    temp_nxk::Matrix{T}
    temp_nxk2::Matrix{T}
    X_I::Matrix{T}
    X_J::Matrix{T}
    X_K::Matrix{T}
    G_J::Matrix{T}
    inter_posI::Vector{Int}
    inter_posJ::Vector{Int}
    r::Int
end

"""Constructor helper for GaussianSketch with preallocated S."""
function GaussianSketch(n::Int, k::Int; T::Type{<:Real}=Float64, c::Real=inv(sqrt(float(k))))
    k <= n || throw(ArgumentError("k must satisfy k <= n"))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    TT = float(T)
    return GaussianSketch{TT}(n, k, TT(c), zeros(TT, n, k))
end

"""Constructor helper for HaarSketch with preallocated S and Gaussian scratch."""
function HaarSketch(n::Int, k::Int; T::Type{<:Real}=Float64, c::Real=sqrt(n / k))
    k <= n || throw(ArgumentError("k must satisfy k <= n"))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    TT = float(T)
    return HaarSketch{TT}(n, k, TT(c), zeros(TT, n, k), zeros(TT, n, k))
end

"""Constructor helper for SubsampleSketch with preallocated sorted index buffer."""
function SubsampleSketch(n::Int, k::Int; c::Real=sqrt(n / k))
    k <= n || throw(ArgumentError("k must satisfy k <= n"))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    return SubsampleSketch(n, k, float(c), Vector{Int}(undef, k))
end

"""Constructor helper for SVRGEstimator with allocated snapshot buffers."""
function SVRGEstimator(Tperiod::Int, n::Int, p::Int; T::Type{<:Real}=Float64)
    Tperiod >= 1 || throw(ArgumentError("SVRG period T must be >= 1"))
    TT = float(T)
    return SVRGEstimator{TT}(
        Tperiod,
        zeros(TT, n, p),
        zeros(TT, n, p),
        zeros(TT, n, p),
    )
end

"""
Initialize `State` for a problem and initial point.

All buffers are allocated once and reused in hot loops.
"""
function init_state(prob, X0::AbstractMatrix{T}, sketch_kind, estimator_kind; k::Int=max(1, min(size(X0)...))) where {T<:Real}
    n0 = size(X0, 1)
    p0 = size(X0, 2)
    n(prob) == n0 || throw(ArgumentError("X0 row size must match n(prob)"))
    p(prob) == p0 || throw(ArgumentError("X0 col size must match p(prob)"))

    kk = max(1, min(k, n0))
    Ip = Matrix{T}(I, p0, p0)

    return State{T}(
        Matrix{T}(X0),
        zeros(T, n0, p0),
        zeros(T, n0, p0),
        zeros(T, n0, p0),
        zeros(T, n0, p0),
        zeros(T, n0, p0),
        zeros(T, n0, p0),
        zeros(T, n0, p0),
        zeros(T, p0, p0),
        zeros(T, p0, p0),
        zeros(T, p0, p0),
        Ip,
        zeros(T, kk, p0),
        zeros(T, kk, p0),
        zeros(T, kk, p0),
        zeros(T, kk, kk),
        zeros(T, n0, kk),
        zeros(T, n0, kk),
        zeros(T, kk, p0),
        zeros(T, kk, p0),
        zeros(T, kk, p0),
        zeros(T, kk, p0),
        zeros(Int, kk),
        zeros(Int, kk),
        0,
    )
end
