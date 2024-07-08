function prune_weights_first!(w, kvecs, inds)
    # choose first kernel vector 
    kvec = first(kvecs)
    if length(kvec) == length(w)
        kvec = view(kvec, inds)
    end
    alphap = Inf
    alphan = Inf
    for (i,k) in enumerate(kvec)
        if k > 0
            alphap = min(alphap, w[inds[i]] / k)
        elseif k < 0
            alphan = max(alphan, w[inds[i]] / k)
        end
    end

    alpha = abs(alphan) < abs(alphap) ? alphan : alphap

    for i in eachindex(kvec)
        w[inds[i]] -= alpha * kvec[i]
    end
end

function prune_weights_minabs!(w, kvecs, inds)
    # choose first kernel vector 
    chosenkvec = nothing
    minabsalpha = Inf
    for kvec in kvecs
        if length(kvec) == length(w)
            kvec = view(kvec, inds)
        end
        # for subtracting the kernel vector
        idp = findall(@. kvec > 0)
        alphap = Inf
        alphan = Inf
        for (i,k) in enumerate(kvec)
            if k > 0
                alphap = min(alphap, w[inds[i]] / k)
            elseif k < 0
                alphan = max(alphan, w[inds[i]] / k)
            end
        end

        alpha = abs(alphan) < abs(alphap) ? alphan : alphap

        if abs(alpha) < minabsalpha
            chosenkvec = kvec
            minabsalpha = alpha
        end
    end

    for i in eachindex(chosenkvec)
        w[inds[i]] -= minabsalpha * chosenkvec[i]
    end
end