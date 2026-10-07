function partition_into_k(M, k)
    groups = Vector{UnitRange{Int}}(undef, k)
    return partition_into_k!(groups, M, k)
end

function partition_into_k!(groups, M, k)
    div, rem = divrem(M, k)
    for i in 1:k
        start = (i-1)*div + min(i-1, rem) + 1
        stop = i*div + min(i, rem)
        groups[i] = start:stop
    end
    return groups
end

"""
`fast_caratheodory(V, w[, frac=e; kwargs...])`

Prune the weighted rows of `V` using an iterative, buffer-reusing variant of
Carathéodory pruning. `V` is an ``M \\times N`` matrix whose rows correspond to
the entries of `w`. A transposed input is also accepted when `length(w)` equals
the number of columns of `V`, following the convention of
[`caratheodory_pruning`](@ref).

At each level, the rows are partitioned into `k = ceil(frac * N)` contiguous
groups. Each group is replaced by its weighted moment row, its group weights
are pruned with [`caratheodory_pruning`](@ref), and the rows belonging to the
surviving groups are compacted in place. Once at most `k` rows remain, they are
pruned directly. The intermediate matrix, weight, index, and group buffers are
allocated once and reused across levels.

`frac` must be greater than one. Keyword arguments are forwarded to
[`caratheodory_pruning`](@ref).

`fast_caratheodory` does not support [`OnDemandMatrix`](@ref) or
[`OnDemandVector`](@ref). It allocates and compacts full intermediate copies of
the rows and weights, so use it only when `V` and `w` can fit in memory.

Returns `(w_pruned, inds)`, where `w_pruned` has the same length as `w`, is
zero outside `inds`, and preserves the weighted moments up to the numerical
accuracy of the underlying pruning method.

This method is inspired by the fast Carathéodory-set construction in
Maalouf, Jubran, and Feldman, [*Fast and Accurate Least-Mean-Squares
Solvers*](https://proceedings.neurips.cc/paper_files/paper/2019/file/475fbefa9ebfba9233364533aafd02a3-Paper.pdf),
NeurIPS 2019.
"""
function fast_caratheodory(V, w, frac=MathConstants.e; kwargs...)
    if V isa OnDemandMatrix || w isa OnDemandVector
        throw(ArgumentError("fast_caratheodory requires in-memory V and w; OnDemandMatrix and OnDemandVector are not supported"))
    end
    frac > 1 || throw(ArgumentError("frac must be > 1 to guarantee reduction, got $frac"))
    M, N = size(V)
    if M != N && N == length(w)
        V = transpose(V)
        M, N = N, M
    end
    if M != length(w)
        error("Dimension mismatch between V ($M×$N) and w ($(length(w)))")
    end
    k = ceil(Int, frac * N)
    if k >= M
        w_sol, inds = caratheodory_pruning(V, w; kwargs...)
        return w_sol, inds
    end
    # Buffers for the surviving rows: at most N groups of size ≤ cld(M, k)
    # survive the first level, and later levels only shrink this
    max_kept = N * cld(M, k)
    V_kept = similar(V, max_kept, N)
    w_kept = similar(w, max_kept)
    orig_rows = Vector{Int}(undef, max_kept)   # row of V each kept row came from
    # Reused each level: one row per group holding Σ w[i] V[i,:], pruned with unit weights
    V_groups = similar(V, k, N)
    w_groups = ones(eltype(w), k)
    groups = Vector{UnitRange{Int}}(undef, k)
    # First level reads from the inputs, later levels compact the buffers in place
    n_rows = _prune_level!(V_kept, w_kept, orig_rows, V, w, 1:M, M, V_groups, w_groups, groups; kwargs...)
    while k < n_rows
        n_rows = _prune_level!(V_kept, w_kept, orig_rows, V_kept, w_kept, orig_rows, n_rows,
                               V_groups, w_groups, groups; kwargs...)
    end
    # Few enough rows left to prune directly
    w_last, kept_last = caratheodory_pruning(
        view(V_kept, 1:n_rows, :), 
        view(w_kept, 1:n_rows); 
        kwargs...
    )
    # Scatter the final weights back to the original rows of V
    inds = orig_rows[kept_last]
    w_sol = zeros(eltype(w_last), M)
    for (orig_row, kept_pos) in zip(inds, kept_last)
        w_sol[orig_row] = w_last[kept_pos]
    end
    return w_sol, inds
end

# One level of group pruning: reads the first n_rows rows of (V_in, w_in, rows_in),
# writes surviving rows to the front of (V_kept, w_kept, orig_rows), returns the new
# row count. Safe when the inputs alias the outputs, since surviving groups only
# move toward the front.
function _prune_level!(V_kept, w_kept, orig_rows, V_in, w_in, rows_in, n_rows,
                       V_groups, w_groups, groups; kwargs...)
    k, _ = size(V_groups)
    partition_into_k!(groups, n_rows, k)
    # Collapse each group into its weighted moment row
    for (j, g) in enumerate(groups)
        mul!(view(V_groups, j, :), transpose(view(V_in, g, :)), view(w_in, g))
    end
    # Prune groups: at most N survive, each with a multiplier on its weights
    group_mult, kept_groups = caratheodory_pruning(V_groups, w_groups; kwargs...)
    sort!(kept_groups)   # needed so each copy below moves rows forward, never back
    # Copy surviving groups forward, scaling weights by their group's multiplier.
    # Each group's destination ends before its source does, so copying in place
    # never overwrites rows of later groups. Explicit loops avoid the defensive
    # copies broadcasting makes when the inputs alias the outputs.
    n_kept = 0
    for j in kept_groups
        g = groups[j]
        off = n_kept + 1 - first(g)   # shift from source row to destination row
        for i in g
            V_kept[i + off, :] .= view(V_in, i, :)
            w_kept[i + off] = w_in[i] * group_mult[j]
            orig_rows[i + off] = rows_in[i]
        end
        n_kept += length(g)
    end
    return n_kept
end
