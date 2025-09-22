"""
`get_min_alpha_k0(w, kvec, inds)`

Helper method that, given a vector of weights, `w`, a kernel
vector `kvec`, and a vector of indices, `inds`, returns a 2-tuple,
`(alpha,k0)` used for pruning. `alpha` is the smallest magnitude
multiple allowed such that `w = w + alpha * kvec` still has
nonnegative entries, and equals zero at index `k0`.
"""
function get_min_alpha_k0(w, kvec, inds)
    alpha = Inf
    k0 = -1
    for (i,k) in enumerate(kvec)
        if k == 0
            continue
        end
        alphanew = w[inds[i]] / k
        if abs(alphanew) < abs(alpha)
            alpha = alphanew
            k0 = i
        end
    end
    return (alpha, k0)
end

"""
`prune_weights_first!(w, kvecs, inds)`

Takes in a vector of full-length weights, `w`,
a vector of kernel vectors, `kvecs`, and a vector
of indices, `inds`, to which the indices of the
kernel vectors point in the weights.

Takes the first kernel vector, and prunes with that,
using the minimum absolute value multiple needed to
zero one of the weights.
"""
function prune_weights_first!(w, kvecs, inds)
    kvec = first(kvecs)
    if length(kvec) == length(w)
        kvec = view(kvec, inds)
    end

    alpha, k0 = get_min_alpha_k0(w, kvec, inds)

    for i in eachindex(kvec)
        w[inds[i]] -= alpha * kvec[i]
    end
    w[inds[k0]] = 0.0
end

"""
`prune_weights_minabs!(w, kvecs, inds)`

Takes in a vector of full-length weights, `w`,
a vector of kernel vectors, `kvecs`, and a vector
of indices, `inds`, to which the indices of the
kernel vectors point in the weights.

Loops over all kernel vectors, and prunes with the 
vector with the minimum absolute value multiple needed to
zero one of the weights.
"""
function prune_weights_minabs!(w, kvecs, inds)
    chosenkvec = nothing
    chosenk0 = -1
    minabsalpha = Inf
    for kvec in kvecs
        if length(kvec) == length(w)
            kvec = view(kvec, inds)
        end
        
        alpha, k0 = get_min_alpha_k0(w, kvec, inds)

        if abs(alpha) < abs(minabsalpha)
            chosenkvec = kvec
            minabsalpha = alpha
            chosenk0 = k0
        end
    end

    for i in eachindex(chosenkvec)
        w[inds[i]] -= minabsalpha * chosenkvec[i]
    end
    w[inds[chosenk0]] = 0.0
end