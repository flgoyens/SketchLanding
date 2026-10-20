"""
Sample `k` unique indices from `1:n` without replacement and store sorted into `idx`.
"""
function sample_without_replacement_sorted!(idx::Vector{Int}, n::Int, k::Int, rng)
    length(idx) == k || throw(ArgumentError("idx length must equal k"))
    k <= n || throw(ArgumentError("k must satisfy k <= n"))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    perm = randperm(rng, n)
    copyto!(idx, view(perm, 1:k))
    sort!(idx)
    return nothing
end

"""
Compute intersection positions between sorted index vectors `I` and `J`.

Writes positions into `posI` and `posJ`; returns `r`, the number of common entries.
"""
function intersect_positions!(posI::Vector{Int}, posJ::Vector{Int}, I::Vector{Int}, J::Vector{Int})
    i = 1
    j = 1
    r = 0
    nI = length(I)
    nJ = length(J)

    while i <= nI && j <= nJ
        Ii = @inbounds I[i]
        Jj = @inbounds J[j]
        if Ii == Jj
            r += 1
            @inbounds posI[r] = i
            @inbounds posJ[r] = j
            i += 1
            j += 1
        elseif Ii < Jj
            i += 1
        else
            j += 1
        end
    end
    return r
end

"""Return effective step-size for built-in scalar step-size modes."""
function compute_stepsize(stepsize_mode::Symbol, stepsize::Real, it::Int, state)
    if stepsize_mode === :constant
        return stepsize
    elseif stepsize_mode === :normalized
        return stepsize / max(one(float(eltype(state.D))), norm(state.D))
    elseif stepsize_mode === :decreasing
        return stepsize / sqrt(float(it))
    else
        throw(ArgumentError("Unknown built-in stepsize mode: $stepsize_mode"))
    end
end

"""Return effective step-size for an iteration-indexed vector schedule."""
compute_stepsize(::Val{:vector}, stepsize::AbstractVector, it::Int, state) = stepsize[it]

"""Return effective step-size for a user-defined step-size rule."""
compute_stepsize(::Val{:function}, stepsize::Function, it::Int, state) = stepsize(it, state)

"""Copy `src[pos,:]` into contiguous `dst[1:r,:]`."""
function gather_rows_by_positions!(dst::AbstractMatrix, src::AbstractMatrix, pos::AbstractVector{Int}, r::Int)
    @views for t in 1:r
        dst[t, :] .= src[pos[t], :]
    end
    return nothing
end

"""
Copy rows selected by global indices `idx` into `dst` with matching order.

This explicit row loop is allocation-friendly: `dst .= src[idx, :]` first builds
the indexed slice `src[idx, :]` as a temporary matrix, then broadcasts into `dst`.
Here we write directly into `dst` row-by-row, so we avoid that temporary `k×p`
allocation and reduce GC pressure in hot paths.
"""
function gather_rows_by_indices!(dst::AbstractMatrix, src::AbstractMatrix, idx::AbstractVector{Int})
    @views for t in eachindex(idx)
        dst[t, :] .= src[idx[t], :]
    end
    return nothing
end

"""
Copy rows selected by global indices `idx` from a sparse CSC matrix into dense `dst`.

This scans each sparse column against the sorted sampled indices, so the work is
proportional to the sparsity in `src` plus the small sample size.
"""
function gather_rows_by_indices!(dst::AbstractMatrix, src::SparseMatrixCSC, idx::AbstractVector{Int})
    fill!(dst, zero(eltype(dst)))
    ncols = size(src, 2)

    @inbounds for j in 1:ncols
        ptr = src.colptr[j]
        stop = src.colptr[j + 1] - 1
        t = firstindex(idx)
        last_t = lastindex(idx)

        while ptr <= stop && t <= last_t
            row = src.rowval[ptr]
            want = idx[t]
            if row == want
                dst[t, j] = src.nzval[ptr]
                ptr += 1
                t += 1
            elseif row < want
                ptr += 1
            else
                t += 1
            end
        end
    end
    return nothing
end

"""Write `dst[idx,:] .= scale * src[idx,:]` after zeroing `dst`."""
function scatter_scaled_selected_rows!(dst::AbstractMatrix, src::AbstractMatrix, idx::AbstractVector{Int}, scale)
    fill!(dst, zero(eltype(dst)))
    @views for i in idx
        dst[i, :] .= scale .* src[i, :]
    end
    return nothing
end
