"""
Algorithm output container for Sketch Landing runs.
"""
mutable struct AlgOutput
    x_final::Matrix{Float64}
    iterations::Vector{Int}
    costs::Union{Nothing,Vector{Float64}}
    infeasibilities::Union{Nothing, Vector{Float64}}
    gradnorms::Union{Nothing, Vector{Float64}}
    times::Union{Nothing,Vector{Float64}}
    iterates::Union{Nothing, Vector{Matrix{Float64}}}
    step_tangent::Union{Nothing, Vector{Matrix{Float64}}}
    step_normal::Union{Nothing, Vector{Matrix{Float64}}}
    norm_iterates::Union{Nothing, Vector{Float64}}
end

"""
Recording toggles for `sketchLanding`.
"""
struct RecordOptions
    record_cost::Bool
    record_time::Bool
    record_infeasibility::Bool
    record_gradnorm::Bool
    record_iterates::Bool
    record_steps::Bool
    record_norm_iterates::Bool
    function RecordOptions(;
        record_cost=true,
        record_time=true,
        record_infeasibility=false,
        record_gradnorm=false,
        record_iterates=false,
        record_steps=false,
        record_norm_iterates=false,
    )
        new(
            record_cost,
            record_time,
            record_infeasibility,
            record_gradnorm,
            record_iterates,
            record_steps,
            record_norm_iterates,
        )
    end
end

"""
Create RecordOptions from a simple integer history level.

history = 0 : minimal storage, only x_final and iterations (not even costs)
history = 1 : scalar traces only (costs, infeasibilities, gradnorms, times, norm_iterates)
history = 2 : full history (iterates + steps)
"""
function create_recorder(; history::Int=1)
    history >= 0 || throw(ArgumentError("history must be ≥ 0"))
    history <= 2 || throw(ArgumentError("history must be ≤ 2"))

    if history == 0
        return RecordOptions(
            record_cost=false,
            record_time=false,
            record_infeasibility=false,
            record_gradnorm=false,
            record_iterates=false,
            record_steps=false,
            record_norm_iterates=false,
        )
    elseif history == 1
        return RecordOptions(
            record_cost=true,
            record_time=true,
            record_infeasibility=true,
            record_gradnorm=true,
            record_iterates=false,
            record_steps=false,
            record_norm_iterates=true,
        )
    else
        return RecordOptions(
            record_cost=true,
            record_time=true,
            record_infeasibility=true,
            record_gradnorm=true,
            record_iterates=true,
            record_steps=true,
            record_norm_iterates=true,
        )
    end
end

"""
Initialize output buffers with fixed length `maxiter + 1` for state traces (including iteration 0).
"""
function init_output(X0::AbstractMatrix, maxiter::Int, record::RecordOptions)::AlgOutput
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1"))
    trace_len = maxiter + 1

    costs = record.record_cost ? fill(NaN, trace_len) : nothing
    times = record.record_time ? zeros(Float64, trace_len) : nothing
    infeas = record.record_infeasibility ? fill(NaN, trace_len) : nothing
    gnorm = record.record_gradnorm ? fill(NaN, trace_len) : nothing
    iters = record.record_iterates ? Vector{Matrix{Float64}}(undef, trace_len) : nothing
    step_tan = record.record_steps ? Vector{Matrix{Float64}}(undef, maxiter) : nothing
    step_norm = record.record_steps ? Vector{Matrix{Float64}}(undef, maxiter) : nothing
    norm_it = record.record_norm_iterates ? fill(NaN, trace_len) : nothing

    return AlgOutput(
        Matrix{Float64}(X0),
        collect(0:maxiter),
        costs,
        infeas,
        gnorm,
        times,
        iters,
        step_tan,
        step_norm,
        norm_it,
    )
end
