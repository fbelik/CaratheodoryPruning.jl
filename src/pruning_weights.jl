function prune_weights_first!(w, kvecs, inds)
    # choose first kernel vector 
    kvec = first(kvecs)
    if length(kvec) == length(w)
        kvec = view(kvec, inds)
    end
    # for subtracting the kernel vector
    idp = findall(@. kvec > 0)
    alphap, k0p = findmin(view(w, inds[idp]) ./ view(kvec, idp))
    k0p = idp[k0p]

    # for adding the kernel vector
    idn = findall(@. kvec < 0)
    alphan, k0n = findmax(view(w, inds[idn]) ./ view(kvec, idn))
    k0n = idn[k0n]

    alpha, k0 = abs(alphan) < abs(alphap) ? (alphan, k0n) : (alphap, k0p)

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
        alphap, k0p = findmin(view(w, inds[idp]) ./ view(kvec, idp))
        k0p = idp[k0p]

        # for adding the kernel vector
        idn = findall(@. kvec < 0)
        alphan, k0n = findmax(view(w, inds[idn]) ./ view(kvec, idn))
        k0n = idn[k0n]

        alpha, k0 = abs(alphan) < abs(alphap) ? (alphan, k0n) : (alphap, k0p)

        if abs(alpha) < minabsalpha
            chosenkvec = kvec
            minabsalpha = alpha
        end
    end

    @. w[inds] -= minabsalpha * chosenkvec
end