"""Sample sketch randomness in-place."""
sample!(::IdentitySketch, rng) = nothing

function sample!(sk::GaussianSketch, rng)
    randn!(rng, sk.S)
    rmul!(sk.S, sk.c) # S = c*S
    return nothing
end

function sample!(sk::HaarSketch, rng)
    randn!(rng, sk.G)
    Q = Matrix(qr!(sk.G).Q) # n×k matrix with orthonormal columns
    copyto!(sk.S, Q)
    rmul!(sk.S, sk.c) # S = c*Q
    return nothing
end

function sample!(sk::SubsampleSketch, rng)
    sample_without_replacement_sorted!(sk.idx, sk.n, sk.k, rng)
    return nothing
end

"""Apply sketch transpose S' to X and write result into Y."""
StX!(Y, ::IdentitySketch, X) = (copyto!(Y, X); nothing)
StX!(Y, ::IdentitySketch, X, state::State) = (copyto!(Y, X); nothing)

function StX!(Y, sk::DenseMatrixSketch, X)
    mul!(Y, sk.S', X) # Y = S'*X
    return nothing
end

function StX!(Y, sk::DenseMatrixSketch, X, state::State)
    mul!(Y, sk.S', X) # Y = S'*X
    return nothing
end

function StX!(Y, sk::SubsampleSketch, X)
    gather_rows_by_indices!(Y, X, sk.idx) # Y = X[sk.idx,:]
    rmul!(Y, sk.c) # Y = c*X[sk.idx,:]
    return nothing
end

function StX!(Y, sk::SubsampleSketch, X, state::State)
    gather_rows_by_indices!(Y, X, sk.idx) # Y = X[sk.idx,:]
    rmul!(Y, sk.c) # Y = c*X[sk.idx,:]
    return nothing
end

"""Apply B = S*S' to X and write result into Y."""
BX!(Y, ::IdentitySketch, X) = (copyto!(Y, X); nothing)
BX!(Y, ::IdentitySketch, X, state::State) = (copyto!(Y, X); nothing)

function BX!(Y, sk::DenseMatrixSketch, X, state::State)
    mul!(state.SX1, sk.S', X) # SX1 = S'*X
    mul!(Y, sk.S, state.SX1) # Y = S*SX1 = (S*S')*X
    return nothing
end

function BX!(Y, sk::SubsampleSketch, X)
    c2 = sk.c * sk.c
    scatter_scaled_selected_rows!(Y, X, sk.idx, c2)
    return nothing
end

function BX!(Y, sk::SubsampleSketch, X, state::State)
    c2 = sk.c * sk.c
    scatter_scaled_selected_rows!(Y, X, sk.idx, c2)
    return nothing
end

"""Compute M = X'*(B*X) in-place."""
XtBX!(M, ::IdentitySketch, X) = (mul!(M, X', X); nothing) # M = X'*X
XtBX!(M, ::IdentitySketch, X, state::State) = (mul!(M, X', X); nothing) # M = X'*X

# Specialized XtBX for DenseMatrixSketch with preallocated S*X in the state.
#  M = X'*(S*S')*X
function XtBX!(M, sk::DenseMatrixSketch, X, state::State)
    mul!(state.SX1, sk.S', X) # SX1 = S'*X
    mul!(M, state.SX1', state.SX1) # M = SX1'*SX1 = X'*(S*S')*X
    return nothing
end

function XtBX!(M, sk::SubsampleSketch, X)
    c2 = sk.c * sk.c
    XI = @view X[sk.idx, :]
    mul!(M, XI', XI) # M = XI'*XI
    rmul!(M, c2) # M = c2*M
    return nothing
end

function XtBX!(M, sk::SubsampleSketch, X, state::State)
    c2 = sk.c * sk.c
    gather_rows_by_indices!(state.X_I, X, sk.idx)
    XI = @view state.X_I[1:sk.k, :]
    mul!(M, XI', XI) # M = XI'*XI
    rmul!(M, c2) # M = c2*M
    return nothing
end

"""
Compute K = X'*(B1*(B2*X)) using generic sketch operations.
"""
function XtB1B2X!(K, sk1::AbstractSketch, sk2::AbstractSketch, X, state::State)
    BX!(state.BX2, sk2, X, state)
    BX!(state.tmp_nxp, sk1, state.BX2, state)
    mul!(K, X', state.tmp_nxp) # K = X'*(B1*(B2*X))
    return nothing
end

"""
Specialized K = X'*(B1*(B2*X)) for identity sketch.
"""
function XtB1B2X!(K, sk1::IdentitySketch, sk2::IdentitySketch, X, state::State)
    mul!(K, X', X) # K = X'*X
    return nothing
end

"""
Specialized K = X'*(B1*(B2*X)) for subsampling sketches using only intersection.
"""
function XtB1B2X!(K, sk1::SubsampleSketch, sk2::SubsampleSketch, X, state::State)
    r = intersect_positions!(state.inter_posI, state.inter_posJ, sk1.idx, sk2.idx)
    state.r = r

    if r == 0
        fill!(K, zero(eltype(K)))
        return nothing
    end

    @views for t in 1:r
        i = sk1.idx[state.inter_posI[t]]
        state.X_K[t, :] .= X[i, :]
    end

    XK = @view state.X_K[1:r, :]
    mul!(K, XK', XK) # K = XK'*XK
    rmul!(K, (sk1.c * sk1.c) * (sk2.c * sk2.c)) # K = (c1^2*c2^2)*K
    return nothing
end

"""Dense debug representation of B for small checks."""
function dense_B(sk::IdentitySketch, n::Int)
    return Matrix{Float64}(I, n, n)
end

function dense_B(sk::DenseMatrixSketch, n::Int)
    return sk.S * sk.S'
end

function dense_B(sk::SubsampleSketch, n::Int)
    B = zeros(Float64, n, n)
    c2 = sk.c * sk.c
    @inbounds for i in sk.idx
        B[i, i] = c2
    end
    return B
end
