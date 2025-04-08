"""
`caratheodory_pruning(V, w_in, kernel_downdater, prune_weights![; caratheodory_correction=true, progress=false, zero_tol=1e-16, return_error=false, errnorm=norm])`

Base method for Caratheodory pruning of the matrix `V` and weights `w_in`.
Returns a new set of weights, `w`, a set of indices, `inds`, and an error
`err` such that `w` only has nonzero elements at the indices, `inds`, and
- if `size(V,1) > size(V,2)`, `||Vᵀw_in - V[inds,:]ᵀw_in[inds]|| = err ≈ 0`
- if `size(V,1) < size(V,2)`, `||V w_in - V[inds,:] w_in[inds]|| = err ≈ 0`
Note that if `return_error=false` and `caratheodory_correction=false`, the
error is not computed and a default return value of 0.0 is used.

Uses the `kernel_downdater` object to generate kernel vectors for pruning,
and the `prune_weights!` method to prune weights after kernel vectors have
been formed.

If `caratheodory_correction=true`, uses a linear solve at the end to reduce
error in the moments.

If `progress=true`, displays a progress bar.

`zero_tol` determines the tolerance for a weight equaling zero.

If `return_error=true` or `caratheodory_correction=true`, computes the 
corresponding errors in moments, `err`. If both `return_error` and
`caratheodory_correction` are set to `false`, `err` is set to 0.0.

`errornorm` is the method called on the truth moments vs computed moments
to evaluate final error, only used if `caratheodory_correction=true` 
or `return_error=true`. Defaults to LinearAlgebra.jl's norm method.
"""
function caratheodory_pruning(V::MType, w_in::VType, kernel_downdater::KernelDowndater, 
                              prune_weights!::Function; caratheodory_correction::Bool=true, 
                              progress::Bool=false, zero_tol::Real=1e-16, return_error::Bool=false, 
                              errnorm::Function=norm
    ) where MType<:AbstractMatrix{T} where VType<:AbstractVector{T} where T
    
    M, N = size(V)
    if M < N
        V = transpose(V)
        M, N = N, M
    end
    if length(w_in) == N # No pruning to do
        return (w_in, collect(1:N), 0.0)
    elseif length(w_in) != M # Dimension mismatch
        error("Dimension mismatch between V ($M×$N) and w ($(length(w_in)))")
    end
    if isa(V, OnDemandMatrix) && V.cols
        @warn "Performance will be slow with current OnDemandMatrix implementation \n         For better performance, transpose OnDemandMatrix storage"
    end
    w = copy(w_in)
    m = M-N
    ct = 1
    err = 0.0
    if caratheodory_correction || return_error
        η_truth = zeros(N)
    end

    if progress
        pbar = ProgressBar(total=m)
        every = max(1, floor(Int, m / 500))
    end
    while ct <= m
        inds = get_inds(kernel_downdater)
        kvecs = get_kernel_vectors(kernel_downdater)
        prune_weights!(w, kvecs, inds)
        # Find all zero weights
        numzeros = 0
        for ind in inds
            if (ct > m)
                break
            end
            if w[ind] < zero_tol
                numzeros += 1
                if caratheodory_correction || return_error
                    η_truth .+= (w_in[ind] .* view(V, ind, :))
                end
                downdate!(kernel_downdater, ind)
                w[ind] = 0.0
                if isa(w, OnDemandVector)
                    forget!(w, ind)
                end
                ct += 1
                if progress && (ct % every == 1)
                    update(pbar, every)
                end
            end
        end
        if numzeros == 0
            error("Did not prune any weights. Check implementation of kernel downdater or prune_weights!.")
        end
    end
    if progress && (pbar.current < m)
        update(pbar, m - pbar.current)
    end
    inds = get_inds(kernel_downdater)
    if caratheodory_correction || return_error
        η_comp = zeros(N)
        for ind in inds
            η_comp .+= (w[ind] .* view(V, ind, :))
            η_truth .+= (w_in[ind] .* view(V, ind, :))
        end
        err = errnorm(η_comp .- η_truth)
    end
    # Try to correct weights
    if caratheodory_correction
        try
            w_cor = transpose(view(V,inds,1:N)) \ η_truth
            # Check if any negative entries
            minentry = minimum(w_cor)
            if minentry <= 0
                w_cor .*= (w_cor .>= 0)
            end
            η_comp .= 0.0
            for (i,ind) in enumerate(inds)
                η_comp .+= (w_cor[i] .* view(V, ind, :))
            end
            new_err = errnorm(η_comp .- η_truth)
            if new_err < err
                w[inds] .= w_cor
                err = new_err
            end
        catch e
            println("Exception during Caratheodory correction: ")
            println(e)
            if !(isa(e, SingularException))
                throw(e)
            end
        end
    end
    return (w::VType, inds::Vector{Int}, err::Float64)
end

"""
`caratheodory_pruning(V, w_in[; kernel=:GivensUpDown, pruning=:first, caratheodory_correction=true, return_error=false, errnorm=norm, zero_tol=1e-16, progress=false, kernel_kwargs...])`

Helper method for calling the base `caratheodory_pruning` method.

Takes in a symbol for `kernel`, and forms a `KernelDowndater` object depending
on what is passed in. Also passes additional kwargs into the `KernelDowndater`:

Options include `:FullQRDowndater` or `:FullQR`, `:GivensDowndater` or `:Givens`,
`:CholeskyDowndater` or `:Cholesky`, `:FullQRUpDowndater` or `:FullQRUpDown`,
and `:GivensUpDownDater` or `:GivensUpDown`.

Takes in a symbol for `pruning`, and chooses a pruning method depending
on what is passed in. Options are `:first` or `:minabs`.

See the other `caratheodory_pruning` docstring for info on other arguments.
"""
function caratheodory_pruning(V::MType, w_in::VType; kernel::Symbol=:GivensUpDown, 
                              pruning::Symbol=:first, caratheodory_correction::Bool=true, 
                              return_error::Bool=false, errnorm::Function=norm, zero_tol::Real=1e-16, 
                              progress::Bool=false, kernel_kwargs...
    ) where MType<:AbstractMatrix{T} where VType<:AbstractVector{T} where T

    M, N = size(V)
    if M < N
        V = transpose(V)
    end
    kernel_downdater = begin
        if kernel in (:FullQRDowndater, :FullQR)
            FullQRDowndater(V; kernel_kwargs...)
        elseif kernel in (:GivensDowndater, :Givens)
            GivensDowndater(V; kernel_kwargs...)
        elseif kernel in (:CholeskyDowndater, :Cholesky)
            CholeskyDowndater(V; kernel_kwargs...)
        elseif kernel in (:FullQRUpDowndater, :FullQRUpDown)
            FullQRUpDowndater(V; kernel_kwargs...)
        elseif kernel in (:GivensUpDowndater, :GivensUpDown)
            GivensUpDowndater(V; kernel_kwargs...)
        else
            error("Unrecognized kernel choice: $(kernel)")
        end
    end
    prune_weights! = begin
        if pruning == :first
            prune_weights_first!
        elseif pruning == :minabs
            prune_weights_minabs!
        else
            error("Unrecognized pruning choice: $(pruning)")
        end
    end
    return caratheodory_pruning(V, w_in, kernel_downdater, prune_weights!, caratheodory_correction=caratheodory_correction, return_error=return_error, errnorm=errnorm, zero_tol=zero_tol, progress=progress)
end