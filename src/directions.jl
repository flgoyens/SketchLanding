"""
Compute normal component N = (B2*X) * (X'*(B1*X) - I) with AbstractSketch 
(only the identity sketch calls this function right now)
M = (X'*(B1*X) - I)
BX2 = B2*X
N = B2X*M
"""
function normal!(N, sk1::AbstractSketch, sk2::AbstractSketch, X, state::State)
    BX!(state.BX2, sk2, X, state)
    XtBX!(state.M, sk1, X, state) # M = X'*(B1*X)
    state.M .= state.M .- state.Ip # M = X'*(B1*X) - I
    mul!(N, state.BX2, state.M) # N = (B2*X)*K
    return nothing
end

"""
Compute normal component N = (B2*X) * (X'*(B1*X) - I) with DenseMatrixSketch
M = (X'*(B1*X) - I)
SX2 = S2'*X
N = S2*[SX2*M]
"""
function normal!(N, sk1::DenseMatrixSketch, sk2::DenseMatrixSketch, X, state::State)
    StX!(state.SX1, sk1, X, state) # SX1 = S1'*X
    StX!(state.SX2, sk2, X, state) # SX2 = S2'*X
    _normal_body!(N, sk1, sk2, state)
    return nothing
end

# Assumes state.SX1 = S1'*X and state.SX2 = S2'*X are already filled.
function _normal_body!(N, sk1::DenseMatrixSketch, sk2::DenseMatrixSketch, state::State)
    mul!(state.M, state.SX1', state.SX1) # M = X'*(S1*S1'*X)
    state.M .-= state.Ip # M = X'*(B1*X) - I
    mul!(state.temp_kxp, state.SX2, state.M) # temp_kxp = SX2*M
    mul!(N, sk2.S, state.temp_kxp) # N = S2*temp_kxp
    return nothing
end

"""
Compute normal component specialized to two subsampling sketches.

For subsampling, `B2*X` is zero outside sampled rows, so we never form a dense
`n×p` product with `K`. Instead we compute only:

`N[sk2.idx, :] = (sk2.c^2) * X[sk2.idx, :] * K`,
where
`K = X'*(B1*X) - I`.

All non-sampled rows are set to zero explicitly.
"""
function normal!(N, sk1::SubsampleSketch, sk2::SubsampleSketch, X, state::State)
    # 1) Build M = X'*(B1*X) - I.
    # XtBX! for SubsampleSketch already avoids building B1 by gathering sampled rows.
    XtBX!(state.M, sk1, X, state) # state.M = X'*(B1*X)
    state.M .-= state.Ip # state.M = X'*(B1*X) - I

    # 2) Gather only sampled rows from X for sketch 2 into contiguous workspace.
    # This avoids multiplying all n rows; we only touch k rows.
    gather_rows_by_indices!(state.X_J, X, sk2.idx) # X_J = X[idx2,:]
    XJ = @view state.X_J[1:sk2.k, :]

    # 3) Reuse another k×p workspace for sampled-row output:
    # XK = X[sk2.idx, :]*K, then scale by sk2.c^2 coming from B2 = S2*S2'.
    XK = @view state.X_K[1:sk2.k, :]
    mul!(XK, XJ, state.M) # k×p times p×p -> k×p (no allocation)
    rmul!(XK, sk2.c * sk2.c) # XK = (sk2.c^2) * X[sk2.idx,:] * K

    # 4) Materialize final N:
    # - zero everything first (rows not in sk2.idx must be zero)
    # - scatter computed k rows back to their original global indices
    fill!(N, zero(eltype(N)))
    @views @inbounds for t in 1:sk2.k
        N[sk2.idx[t], :] .= XK[t, :]
    end
    return nothing
end

"""
Compute tangent component T = G*K - (B1*X)*(G'*(B2*X))
for AbstractSketch. 
where 
K = X'*(B1*(B2*X)) is pxp
P = G'*(B2*X)
tmp_nxp = (B1*X)*P
T = G*K - tmp_nxp
The product G*K is np^2 here 
B2*X is computed twice
"""
function tangent!(T, sk1::AbstractSketch, sk2::AbstractSketch, X, G, state::State)
    BX!(state.BX1, sk1, X, state) # BX1
    BX!(state.BX2, sk2, X, state) # BX2
    XtB1B2X!(state.K, sk1, sk2, X, state) # K = X'*B1*B2*X
    mul!(state.P, G', state.BX2) # P = G'*(B2*X)
    mul!(T, G, state.K) # T = G * K
    mul!(state.tmp_nxp, state.BX1, state.P) # tmp_nxp = (B1*X)*P
    T .-= state.tmp_nxp # T = G*K - (B1*X)*P
    return nothing
end

"""
Compute tangent component for DenseMatrixSketch (Gaussian, Haar, …) without any
`n×p` by `p×p` products.

With `B1 = S1*S1'`, `B2 = S2*S2'`, `SX1 = S1'X`, `SX2 = S2'X`, and
`SG2 = S2'G`, we use

T = G*(X'B1B2X) - (B1X)*(G'B2X)

and factor it as

T = (G*(S1'X)'*(S1'S2) - S1*(S1'X*(S2'G)')) * S2'X.

All dense products are therefore either `n×p` by `p×k`, `n×k` by `k×k`, or
`n×k` by `k×p`, with `k << p <= n`.
"""
function tangent!(T, sk1::DenseMatrixSketch, sk2::DenseMatrixSketch, X, G, state::State)
    StX!(state.SX1, sk1, X, state) # SX1 = S1'*X
    StX!(state.SX2, sk2, X, state) # SX2 = S2'*X
    _tangent_body!(T, sk1, sk2, G, state)
    return nothing
end

# Assumes state.SX1 = S1'*X and state.SX2 = S2'*X are already filled.
function _tangent_body!(T, sk1::DenseMatrixSketch, sk2::DenseMatrixSketch, G, state::State)
    StX!(state.temp_kxp, sk2, G, state) # temp_kxp = SG2 = S2'*G

    mul!(state.temp_kxk, sk1.S', sk2.S) # temp_kxk = S1'*S2
    mul!(state.temp_nxk, G, state.SX1') # temp_nxk = G*SX1'
    mul!(state.temp_nxk2, state.temp_nxk, state.temp_kxk) # temp_nxk2 = G*SX1'*(S1'*S2)
    mul!(T, state.temp_nxk2, state.SX2) # T = G*SX1'*(S1'*S2)*SX2

    mul!(state.temp_kxk, state.SX1, state.temp_kxp') # temp_kxk = SX1*SG2'
    mul!(state.temp_nxk, sk1.S, state.temp_kxk) # temp_nxk = S1*(SX1*SG2')
    mul!(state.tmp_nxp, state.temp_nxk, state.SX2) # tmp_nxp = S1*(SX1*SG2')*SX2
    T .-= state.tmp_nxp
    return nothing
end

"""
Compute tangent component T = G*K - (B1*X)*(G'*(B2*X)),
where K = X'*(B1*(B2*X)) for SubsampleSketch.
"""
function tangent!(T, sk1::SubsampleSketch, sk2::SubsampleSketch, X, G, state::State)
    c12 = sk1.c * sk1.c
    c22 = sk2.c * sk2.c

    gather_rows_by_indices!(state.X_I, X, sk1.idx) # X_I = X[sk1.idx,:]
    gather_rows_by_indices!(state.X_J, X, sk2.idx) # X_J = X[sk2.idx,:]
    gather_rows_by_indices!(state.G_J, G, sk2.idx) # G_J = G[sk2.idx,:], sparse-aware

    XI = @view state.X_I[1:sk1.k, :]
    XJ = @view state.X_J[1:sk2.k, :]
    GJ = @view state.G_J[1:sk2.k, :]
    fill!(T, zero(eltype(T)))

    r = intersect_positions!(state.inter_posI, state.inter_posJ, sk1.idx, sk2.idx)
    state.r = r

    if r > 0
        @views for t in 1:r
            state.X_K[t, :] .= XI[state.inter_posI[t], :]
        end

        XL = @view state.X_K[1:r, :]
        Y = @view state.temp_nxk[:, 1:r]
        mul!(Y, G, XL') # Y = G*X_L', uses sparse matmul if G is sparse
        mul!(T, Y, XL) # T = (G*X_L')*X_L
        rmul!(T, c12 * c22)
    end

    mul!(state.temp_kxk, XI, GJ') # temp_kxk = X_I*G_J'
    sampled_term = @view state.X_K[1:sk1.k, :]
    mul!(sampled_term, state.temp_kxk, XJ) # sampled_term = (X_I*G_J')*X_J
    rmul!(sampled_term, c12 * c22)

    @views @inbounds for t in 1:sk1.k
        T[sk1.idx[t], :] .-= sampled_term[t, :]
    end

    # Archive of the previous implementation:
    # BX!(state.BX1, sk1, X, state)
    # BX!(state.BX2, sk2, X, state)
    # XtB1B2X!(state.K, sk1, sk2, X, state)
    # mul!(state.P, G', state.BX2) # P = G'*(B2*X)
    #
    # mul!(T, G, state.K) # T = G*K  what are the dimensions here? look at Simon notes and use S,
    # # probably not computing K, but expand S.
    # mul!(state.BX2, state.BX1, state.P) # BX2 = (B1*X)*P
    # T .-= state.BX2
    return nothing
end

"""
Compute sketched direction for DenseMatrixSketch (Gaussian, Haar, …), computing
SX1 = S1'X and SX2 = S2'X once and sharing them between the tangent and normal
components.
"""
function sketched_direction_from_gradient!(D, sk1::DenseMatrixSketch, sk2::DenseMatrixSketch, X, G, state::State; ωT, ωN)
    compute_T = !iszero(ωT)
    compute_N = !iszero(ωN)
    if compute_T || compute_N
        StX!(state.SX1, sk1, X, state) # SX1 = S1'*X  (shared)
        StX!(state.SX2, sk2, X, state) # SX2 = S2'*X  (shared)
    end
    if compute_T
        _tangent_body!(state.Tstep, sk1, sk2, G, state)
    else
        fill!(state.Tstep, zero(eltype(state.Tstep)))
    end
    if compute_N
        _normal_body!(state.Nstep, sk1, sk2, state)
    else
        fill!(state.Nstep, zero(eltype(state.Nstep)))
    end
    rmul!(state.Nstep, ωN) # Nstep = ωN*N
    rmul!(state.Tstep, ωT) # Tstep = ωT*T
    copyto!(D, state.Nstep)
    D .+= state.Tstep
    return nothing
end

"""Compute sketched direction D = ωT*T + ωN*N given gradient G already filled."""
function sketched_direction_from_gradient!(D, sk1, sk2, X, G, state::State; ωT, ωN)
    if !iszero(ωT)
        tangent!(state.Tstep, sk1, sk2, X, G, state)
    else
        fill!(state.Tstep, zero(eltype(state.Tstep)))
    end
    if !iszero(ωN)
        normal!(state.Nstep, sk1, sk2, X, state)
    else
        fill!(state.Nstep, zero(eltype(state.Nstep)))
    end
    rmul!(state.Nstep, ωN) # Nstep = ωN*N
    rmul!(state.Tstep, ωT) # Tstep = ωT*T
    copyto!(D, state.Nstep)
    D .+= state.Tstep
    return nothing
end

"""Compute deterministic full direction using identity sketches and deterministic gradient."""
function full_direction!(D, prob::AbstractProblem, X, state::State; ωT, ωN)
    # Skip the gradient when ωT == 0: only the normal component is used, which
    # does not depend on G.
    if !iszero(ωT)
        grad!(state.G, prob, X)
    end
    sk = IdentitySketch()
    sketched_direction_from_gradient!(D, sk, sk, X, state.G, state; ωT=ωT, ωN=ωN)
    return nothing
end

"""
Plain estimator direction.

Uses stochastic gradient `stoch_grad!`; fallback is deterministic `grad!`.
"""
function direction!(D, sk1, sk2, ::PlainEstimator, prob::AbstractProblem, X, state::State; ωT, ωN, ξ, rng)
    # Skip the gradient when ωT == 0: only the normal component is used, which
    # does not depend on G.
    if !iszero(ωT)
        stoch_grad!(state.G, prob, X, ξ)
    end
    sketched_direction_from_gradient!(D, sk1, sk2, X, state.G, state; ωT=ωT, ωN=ωN)
    return nothing
end

"""
SVRG direction:
D = Dsketch(X) - Dsketch(Y) + Dfull(Y)

By default `Dsketch(Y)` uses deterministic gradient at snapshot (`snapshot_stochastic=false`).
"""
function direction!(D, sk1, sk2, svrg::SVRGEstimator, prob::AbstractProblem, X, state::State;
    ωT,
    ωN,
    ξ,
    rng,
    snapshot_stochastic::Bool=false,
)
    # Skip both gradients when ωT == 0: every direction below uses only the
    # normal component, which does not depend on G.
    compute_grad = !iszero(ωT)

    if compute_grad
        stoch_grad!(state.G, prob, X, ξ)
    end
    sketched_direction_from_gradient!(state.D, sk1, sk2, X, state.G, state; ωT=ωT, ωN=ωN)

    if compute_grad
        if snapshot_stochastic
            stoch_grad!(state.G, prob, svrg.snapshotX, ξ)
        else
            grad!(state.G, prob, svrg.snapshotX)
        end
    end
    sketched_direction_from_gradient!(svrg.snapshotDsketch, sk1, sk2, svrg.snapshotX, state.G, state; ωT=ωT, ωN=ωN)

    copyto!(D, state.D)
    D .-= svrg.snapshotDsketch
    D .+= svrg.snapshotDfull
    return nothing
end
