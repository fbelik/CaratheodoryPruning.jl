function prune_weights_first!(w, kvecs, inds)
    # choose first kernel vector 
    kvec = first(kvecs)
    if length(kvec) == length(w)
        kvec = view(kvec, inds)
    end
    # for subtracting the kernel vector
    idp = findall(@. kvec > 0)
    alphap = Inf
    if length(idp) > 0
        alphap = findmin(view(w, inds[idp]) ./ view(kvec, idp))[1]
    end

    # for adding the kernel vector
    idn = findall(@. kvec < 0)
    alphan = -Inf
    if length(idn) > 0
        alphan = findmax(view(w, inds[idn]) ./ view(kvec, idn))[1]
    end

    alpha = abs(alphan) < abs(alphap) ? alphan : alphap

    @. w[inds] -= alpha * kvec
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
        if length(idp) > 0
            alphap = findmin(view(w, inds[idp]) ./ view(kvec, idp))[1]
        end

        # for adding the kernel vector
        idn = findall(@. kvec < 0)
        alphan = -Inf
        if length(idn) > 0
            alphan = findmax(view(w, inds[idn]) ./ view(kvec, idn))[1]
        end

        alpha = abs(alphan) < abs(alphap) ? alphan : alphap

        if abs(alpha) < minabsalpha
            chosenkvec = kvec
            minabsalpha = alpha
        end
    end

    @. w[inds] -= minabsalpha * chosenkvec
end