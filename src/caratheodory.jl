function caratheodory_pruning(V, w_in, kernel_downdater::KernelDowndater, prune_weights!::Function, caratheodory_correction=false; zero_tol=1e-16, return_errors=false, errnorm=norm)
    
    if length(w_in) <= size(V, 2)
        return w_in, eachindex(w_in)
    end
    w = copy(w_in)
    M, N = size(V)
    m = M-N
    inds = collect(1:M)
    ct = 1

    if return_errors
        Vtw = V'w
        errors = zeros(m)
    end

    while ct <= m
        kvecs = get_kernel_vectors(kernel_downdater)
        prune_weights!(w, kvecs, inds)
        # Find all zero weights
        allzeros = findall(x -> abs(x) <= zero_tol, view(w, inds))
        for k0 in allzeros
            downdate!(kernel_downdater, inds[k0])
            deleteat!(inds, k0)
            if return_errors
                errors[ct] = errnorm(V'w .- Vtw)
            end
            ct += 1
        end
    end
    # Try to correct weights
    if caratheodory_correction
        try
            w_cor = view(V,inds,1:N)' \ Vtw
            # Check if any negative entries
            minentry = minimum(w_cor)
            # println("Min entry: $(minentry)")
            if minentry < 0
                w_cor .*= (w_cor .>= 0)
            end
            w[inds] .= w_cor
        catch e
            println("Exception: ")
            println(e)
            if !(isa(e, SingularException))
                throw(e)
            end
        end
    end
    if return_errors
        return w, inds, errors
    else
        return w, inds
    end
end

function caratheodory_pruning(V, w_in, caratheodory_correction=false; kernel=:CholeskyDowndater, pruning=:first, return_errors=false, errnorm=norm, zero_tol=1e-16, kernel_kwargs...)
    kernel_downdater = begin
        if kernel in (:FullQRDowndater, :FullQR)
            FullQRDowndater(V; kernel_kwargs...)
        elseif kernel in (:GivensQRDowndater, :GivensQR)
            GivensQRDowndater(V; kernel_kwargs...)
        elseif kernel in (:CholeskyDowndater, :Cholesky)
            CholeskyDowndater(V; kernel_kwargs...)
        elseif kernel in (:CholeskyUpDowndater, :CholeskyUpDown)
            CholeskyUpDowndater(V; kernel_kwargs...)
        else
            error("Unrecognized kernel choice: $(kernel)")
        end
    end
    prune_weights! = begin
        if pruning == :first
            prune_weights_first!
        elseif pruning == :minabs
            prune_weights_minabs!
        end
    end
    return caratheodory_pruning(V, w_in, kernel_downdater, prune_weights!, caratheodory_correction, return_errors=return_errors, errnorm=errnorm, zero_tol=zero_tol)
end