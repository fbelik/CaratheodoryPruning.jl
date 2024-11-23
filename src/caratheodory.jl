"""
`caratheodory_pruning(V, w_in, kernel_downdater, prune_weights![; caratheodory_correction=false, progress=false, zero_tol=1e-16, return_errors=false, errnorm=norm])`

Base method for Caratheodory pruning of the matrix `V` and weights `w_in`.
Returns a new set of weights, `w`, and a set of indices, `inds`, such that
`w_in` only has nonzero elements at the indices, `inds`, and
- if `size(V,1) > size(V,2)`, `Vᵀw_in - V[inds,:]ᵀw_in[inds] ≈ 0`
- if `size(V,1) < size(V,2)`, `V w_in - V[inds,:] w_in[inds] ≈ 0`

Uses the `kernel_downdater` object to generate kernel vectors for pruning,
and the `prune_weights!` method to prune weights after kernel vectors have
been formed.

If `caratheodory_correction=true`, then uses a linear solve at the end to reduce
error in the moments.

If `progress=true`, displays a progress bar.

`zero_tol` determines the tolerance for a weight equaling zero.

If `return_errors=true`, returns an additional vector of moment errors throughout
the procedure.

`errornorm` is the method called on `Vᵀw_in - V[inds,:]ᵀw_in[inds]` or 
`V w_in - V[inds,:] w_in[inds]` to evaluate errors, only used if 
`caratheodory_correction=true` or `return_errors=true`. Defaults 
to LinearAlgebra.jl's norm method.
"""
function caratheodory_pruning(V::AbstractMatrix, w_in::AbstractVector, kernel_downdater::KernelDowndater, prune_weights!::Function; caratheodory_correction=false, progress=false, zero_tol=1e-16, return_errors=false, errnorm=norm)
    
    if length(w_in) <= size(V, 2)
        return w_in, eachindex(w_in)
    end
    w = copy(w_in)
    M, N = size(V)
    if M < N
        V = transpose(V)
        M, N = N, M
    end
    m = M-N
    ct = 1
    
    if caratheodory_correction || return_errors
        Vtw = transpose(V)*w
    end

    if return_errors
        errors = zeros(m)
    end
    if progress
        pbar = ProgressBar(total=m)
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
                downdate!(kernel_downdater, ind)
                w[ind] = 0.0
                if isa(w, OnDemandVector)
                    forget!(w, ind)
                end
                if return_errors
                    errors[ct] = errnorm(transpose(V)*w .- Vtw)
                end
                ct += 1
                if progress
                    update(pbar)
                end
            end
        end
        if numzeros == 0
            error("Did not prune any weights. Check implementation of kernel downdater or prune_weights!.")
        end
    end
    # Try to correct weights
    if caratheodory_correction
        try
            inds = get_inds(kernel_downdater)
            w_cor = view(V,inds,1:N)' \ Vtw
            # Check if any negative entries
            minentry = minimum(w_cor)
            if minentry <= 0
                w_cor .*= (w_cor .>= 0)
            end
            Vt = transpose(view(V,inds,:))
            error = errnorm(Vt * view(w,inds) .- Vtw)
            corrected_error = errnorm(Vt * w_cor .- Vtw)
            if corrected_error < error
                w[inds] .= w_cor
                error = corrected_error
            end
            
            if return_errors
                push!(errors, error)
            end
        catch e
            println("Exception: ")
            println(e)
            if !(isa(e, SingularException))
                throw(e)
            end
        end
    end
    inds = get_inds(kernel_downdater)
    if return_errors
        return w, inds, errors
    else
        return w, inds
    end
end

"""
`caratheodory_pruning(V, w_in[; kernel=:GivensUpDown, pruning=:first, caratheodory_correction=false, return_errors=false, errnorm=norm, zero_tol=1e-16, progress=false, kernel_kwargs...])`

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
function caratheodory_pruning(V::AbstractMatrix, w_in::AbstractVector; kernel=:GivensUpDown, pruning=:first, caratheodory_correction=false, return_errors=false, errnorm=norm, zero_tol=1e-16, progress=false, kernel_kwargs...)
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
    return caratheodory_pruning(V, w_in, kernel_downdater, prune_weights!, caratheodory_correction=caratheodory_correction, return_errors=return_errors, errnorm=errnorm, zero_tol=zero_tol, progress=progress)
end