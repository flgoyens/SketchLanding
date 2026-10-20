"""Update estimator state before direction evaluation."""
update_estimator!(::PlainEstimator, prob, X, it::Int, state::State, rng; ωT=one(eltype(X)), ωN=one(eltype(X))) = nothing

"""
SVRG snapshot update.

Refresh snapshot point and full direction at iterations 1, 1+T, 1+2T, …
(equivalently, whenever `(it - 1) % T == 0`). This is correct for every T ≥ 1;
in particular T = 1 refreshes every iteration, recovering deterministic Landing.
"""
function update_estimator!(svrg::SVRGEstimator, prob::AbstractProblem, X, it::Int, state::State, rng; ωT, ωN)
    if (it - 1) % svrg.T == 0
        copyto!(svrg.snapshotX, X)
        full_direction!(svrg.snapshotDfull, prob, svrg.snapshotX, state; ωT=ωT, ωN=ωN)
    end
    return nothing
end
