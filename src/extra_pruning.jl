"""
`extra_pruning!(V, w, inds[, zero_tol=1e-16, sval_tol=1e-15])`

Takes a matrix, `V`, and a vector of weights, `w`, and corresponding
indices, `inds`, and prunes such that reduces the number of indices, 
modifying the weights, `w`, and the indices, `inds`, in place maintaining
the moments given by `η = V[inds,:]ᵀw[inds]`.

It only halts when `σ_min / σ_max > sval_tol` or when 
`σ_min * α > zero_tol` where `σ_min` is the minimum singular value of 
`V[inds,:]`, `σ_max` is the maximum singular value of `V[inds,:]`, and 
`α` is the multiple of the kernel vector that is used to prune the weights.
"""
function extra_pruning!(V, w, inds, zero_tol=1e-16, sval_tol=1e-15)
    if isa(inds, AbstractRange)
        inds = collect(inds)
    end
    while true
        _,S,Vmat = svd(view(V, inds, :))
        kvec = Vmat[:,end]

        if (S[end] / S[1]) > sval_tol
            break
        end
        alphan, k0n, alphap, k0p = get_alpha_k0s(w, kvec, inds)

        alpha, k0 = abs(alphan) < abs(alphap) ? (alphan, k0n) : (alphap, k0p)

        if (S[end] * alpha) > zero_tol
            break
        end

        inds_delete = [false for _ in inds]
        for i in eachindex(kvec)
            w[inds[i]] -= alpha * kvec[i]
            if (i == k0) || (abs(w[inds[i]]) < zero_tol)
                w[inds[i]] = 0.0
                if isa(w, OnDemandVector)
                    forget!(w, inds[i])
                end
                if isa(V, OnDemandMatrix)
                    forget!(V, inds[i])
                end
                inds_delete[i] = true
            end
        end

        deleteat!(inds, inds_delete)
    end
    return w, inds
end