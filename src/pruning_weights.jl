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
    alphap = Inf
    k0p = -1
    alphan = -Inf
    k0n = -1
    for (i,k) in enumerate(kvec)
        alphanew = w[inds[i]] / k
        if alphanew != 0.0
            if (k > 0) && (alphanew < alphap)
                alphap = alphanew
                k0p = i
            elseif (k < 0) && (alphanew > alphan)
                alphan = alphanew
                k0n = i
            end
        end
    end

    alpha, k0 = abs(alphan) < abs(alphap) ? (alphan, k0n) : (alphap, k0p)

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
        alphap = Inf
        k0p = -1
        alphan = -Inf
        k0n = -1
        for (i,k) in enumerate(kvec)
            alphanew = w[inds[i]] / k
            if alphanew != 0.0
                if (k > 0) && (alphanew < alphap)
                    alphap = alphanew
                    k0p = i
                elseif (k < 0) && (alphanew > alphan)
                    alphan = alphanew
                    k0n = i
                end
            end
        end

        alpha, k0 = abs(alphan) < abs(alphap) ? (alphan, k0n) : (alphap, k0p)
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