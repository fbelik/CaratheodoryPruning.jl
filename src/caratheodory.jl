"""
`caratheodory_pruning(V, w_in, kernel_downdater, prune_weights![; caratheodory_correction=true, progress=false, zero_tol=1e-16, return_error=false, errnorm=norm, extra_pruning=false, sval_tol=1e-15])`

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

If `extra_pruning=true`, additional pruning is attempted after the initial
pruning to find a further reduced rule (less points than number of moments).
This pruning is determined by `sval_tol` and `zero_tol` as it checks if the
pruned Vandermonde matrix has singular values close to zero.
"""
function caratheodory_pruning(V, w_in, kernel_downdater::KernelDowndater, 
                              prune_weights!::Function; caratheodory_correction=true, 
                              progress::Bool=false, zero_tol=1e-16, return_error=false, 
                              errnorm::Function=norm, extra_pruning=false, sval_tol=1e-15) 
    
    M, N = size(V)
    if M < N
        V = transpose(V)
        M, N = N, M
    end
    if length(w_in) != M # Dimension mismatch
        error("Dimension mismatch between V ($M×$N) and w ($(length(w_in)))")
    end
    if isa(V, OnDemandMatrix) && V.cols
        msg = "Performance will be slow with current OnDemandMatrix implementation \n"
        msg *= "         For better performance, transpose OnDemandMatrix storage"
        @warn msg
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
    # Compute error
    if caratheodory_correction || return_error
        η_comp = zeros(N)
        for ind in inds
            η_comp .+= (w[ind] .* view(V, ind, :))
            η_truth .+= (w_in[ind] .* view(V, ind, :))
        end
        err = errnorm(η_comp .- η_truth)
    end
    # Prune extra
    if extra_pruning
        err += extra_pruning!(V, w, inds, zero_tol, sval_tol)
    end
    # Try to correct weights
    if caratheodory_correction
        viewVt = transpose(view(V, inds, 1:N))
        w_cor = similar(view(w, inds))
        try
            w_cor .= viewVt \ η_truth
        catch e
            if isa(e, SingularException)
                w_cor .= pinv(viewVt) * η_truth
            end
            throw(e)
        end
        # Check if any negative entries
        minentry = minimum(w_cor)
        posinds = (w_cor .> 0)
        if minentry <= 0
            w_cor .*= posinds
        end
        η_comp .= 0.0
        for (i,ind) in enumerate(inds)
            η_comp .+= (w_cor[i] .* view(V, ind, :))
        end
        new_err = errnorm(η_comp .- η_truth)
        if new_err < err
            w[inds] .= w_cor
            deleteat!(inds, (.! posinds))
            err = new_err
        end
    end
    return w, inds, err
end

"""
`caratheodory_pruning(V, w_in[; kernel=GivensUpDownDater, pruning=:first, caratheodory_correction=true, return_error=false, errnorm=norm, zero_tol=1e-16, progress=false, kernel_kwargs...])`

Helper method for calling the base `caratheodory_pruning` method.

Takes in a method, `kernel`, to form a `KernelDowndater` object.
Options include `FullQRDowndater`, `GivensDowndater`, `CholeskyDowndater`, 
`FullQRUpDowndater`, and `GivensUpDownDater`.

Additional kwargs are passed into the `KernelDowndater` constructor.

Takes in a pruning method for `pruning`, current implemented options
include `prune_weights_first!` and `prune_weights_minabs!`. These methods
must use a set of kernel vectors to prune elements of the weight vector.

See the other `caratheodory_pruning` docstring for info on other arguments.
"""
function caratheodory_pruning(V, w_in; kernel=GivensUpDowndater, 
                              pruning=prune_weights_first!, caratheodory_correction=true, 
                              return_error=false, errnorm=norm, zero_tol=1e-16, 
                              progress=false, extra_pruning=false, sval_tol=1e-15,
                              kernel_kwargs...) 

    M, N = size(V)
    if M < N
        V = transpose(V)
    end
    kernel_downdater = kernel(V; kernel_kwargs...)
    prune_weights! = pruning
    return caratheodory_pruning(V, w_in, kernel_downdater, prune_weights!, caratheodory_correction=caratheodory_correction, 
                                return_error=return_error, errnorm=errnorm, zero_tol=zero_tol, progress=progress, 
                                extra_pruning=extra_pruning, sval_tol=sval_tol)
end