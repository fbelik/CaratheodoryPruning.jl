abstract type KernelDowndater end

# Required methods for KernelDowndater objects
"""
`get_inds(kd::KernelDowndater)`

Method that must be implemented for abstract type `KernelDowndater`.

Returns an abstract vector of indices, between 1 and the number of 
rows of the matrix `V`, for which when `get_kernel_vectors(kd)` is called,
each kernel vector, `kvec`, is only assumed to be filled at the indices, i.e.,
`transpose(V[inds,:]) * kvec = [0,0,...,0]`.
Cannot contain any indices that were previously removed by `downdate!`.
"""
function get_inds(kd::KernelDowndater)
    error("Must implement inds for KernelDowndater")
end

"""
`get_kernel_vectors(kd::KernelDowndater)`

Method that must be implemented for abstract type `KernelDowndater`.

Returns an abstract vector of kernel vectors, such that after `inds = get_inds(kd)`,
and `kvecs = get_kernel_vectors(kd)`, for each `kvec` in `kvecs`,
`transpose(V[inds,:]) * kvec = [0,0,...,0]`.
"""
function get_kernel_vectors(kd::KernelDowndater)
    error("Must implement get_kernel_vectors for KernelDowndater")
end

"""
`downdate!(kd::KernelDowndater, idx::Int)`

Method that must be implemented for abstract type `KernelDowndater`.

Must pass in an index `idx` in `get_inds(kd)`. This method downdates
`kd` such that any new kernel vectors returned will not include
the row index `idx` of `V`.
"""
function downdate!(kd::KernelDowndater, idx::Int) 
    error("Must implement downdate! for KernelDowndater")
end

"""
`FullQRDowndater`

A mutable struct to hold the `Q`-factor of the QR decomposition
of the matrix `V[inds,:]` for generating vectors in the kernel 
of its transpose. Downdates by forming a new QR factor, which takes
`O(M N²)` flops where `V` is an `M x N` matrix.

Form with `FullQRDowndater(V[; k=1])` where `k` is the (maximum) number
of kernel vectors returned each time `get_kernel_vectors` is called.
"""
mutable struct FullQRDowndater <: KernelDowndater
    V::AbstractMatrix
    Q::Union{AbstractMatrix,QRCompactWYQ}
    inds::AbstractVector
    N::Int
    k::Int
    function FullQRDowndater(V::AbstractMatrix; k=1)
        M, N = size(V)
        m = M - N
        if k > m
            k = m
        end
        ct = 1
        inds = collect(1:M)
        Q,_ = qr(V)
        return new(V, Q, inds, N, k)
    end
end

function get_inds(kd::FullQRDowndater)
    return kd.inds
end

function get_kernel_vectors(kd::FullQRDowndater)
    startidx = max(size(kd.Q, 2)-kd.k+1, kd.N+1)
    kvecs = eachcol(kd.Q[:, startidx:size(kd.Q, 2)])
    return kvecs
end

function downdate!(kd::FullQRDowndater, idx::Int)
    filter!(!=(idx), kd.inds)
    if size(kd.Q,1) == (size(kd.V, 2) + 1)
        # Don't need to update anymore
        return
    end
    newQ, _ = qr(view(kd.V, kd.inds, 1:kd.N))
    kd.Q = newQ
end

"""
`GivensDowndater`

A mutable struct to hold the `Q`-factor of the QR decomposition
of the matrix `V[inds,:]` for generating vectors in the kernel 
of its transpose. Downdates by applying Givens rotations to the
old, full, `Q`-factor, which takes `O(M²)` flops where `V` is an `M x N` matrix.

Form with `GivensDowndater(V[; k=1])` where `k` is the (maximum) number
of kernel vectors returned each time `get_kernel_vectors` is called.
"""
mutable struct GivensDowndater <: KernelDowndater
    V::AbstractMatrix
    Q::Union{AbstractMatrix,QRCompactWYQ}
    inds::AbstractVector
    N::Int
    k::Int
    function GivensDowndater(V::AbstractMatrix; k=1)
        M, N = size(V)
        m = M - N
        if k > m
            k = m
        end
        ct = 1
        # Allocate arrays
        inds = collect(1:M)
        Q,_ = qr(V)
        Q = Q[1:M,1:M]
        return new(V, Q, inds, N, k)
    end
end

function get_inds(kd::GivensDowndater)
    return kd.inds
end

function get_kernel_vectors(kd::GivensDowndater)
    inds = kd.inds
    startidx = max(length(inds)-kd.k+1, kd.N+1)
    kvecs = eachcol(view(kd.Q, inds, view(inds, startidx:length(inds))))
    return kvecs
end

function downdate!(kd::GivensDowndater, idx::Int)
    inds = kd.inds
    mididx = findfirst(==(idx), kd.inds)
    if length(inds) == (size(kd.V, 2) + 1)
        # Don't need to update anymore
        deleteat!(inds, mididx)
        return
    end
    q = view(kd.Q, idx, inds)
    r = q[end]
    for i in length(q):-1:(mididx+1)
        G, r = givens(q[i-1], r, i-1, i)
        rmul!(view(kd.Q,inds,inds), G')
    end
    for i in (mididx-1):-1:1
        G, r = givens(r, q[i], mididx, i)
        rmul!(view(kd.Q,inds,inds), G')
    end
    deleteat!(inds, mididx)
end

"""
`CholeskyDowndater`

A mutable struct to hold the `Q`-factor of the QR decomposition
of the matrix `V[inds,:]` for generating vectors in the kernel 
of its transpose. Downdates by reorthogonalizing the old `Q`-factor,
with a row removed, by multiplication by the inverse transpose of its
Cholesky factor, which takes `O(N³ + MN)` flops where `V` is an `M x N` matrix.

Form with `CholeskyDowndater(V[; k=1, pct_full_qr=10.0, SM_tol=1e-6, full_Q=false)])`.
`k` is the (maximum) number of kernel vectors returned each time 
`get_kernel_vectors` is called. `pct_full_qr` is the percentage (between 0 and 100),
of times, logarithmically spaced, that a full QR reset will be done to prevent
accumulation of error. `SM_tol` is a tolerance on the denominator of the Sherman Morrison
formula to prevent error from division close to zero. `full_Q` determines
whether or not the full Q matrix is updated or just its Cholesky factor; if set
to `true`, will take `O(N³ + MN²)` flops instead of `O(N³ + MN)`.

From testing, seems to have minimal error accumulation if `pct_full_qr ≥ 10.0`.
"""
mutable struct CholeskyDowndater <: KernelDowndater
    V::AbstractMatrix
    Q::AbstractMatrix
    D::AbstractMatrix
    x::AbstractVector
    C::AbstractMatrix
    inds::AbstractVector
    ct::Int
    m::Int
    N::Int
    k::Int
    full_Q::Bool
    full_forced_inds::AbstractVector{<:Int}
    SM_tol::Float64
    function CholeskyDowndater(V::AbstractMatrix; k=1, pct_full_qr=10.0, SM_tol=1e-6, full_Q=false)
        M, N = size(V)
        m = M - N
        if k > m
            k = m
        end
        ct = 1
        # Allocate arrays
        inds = collect(1:M)
        Q,_ = qr(V)
        Q = Q[1:M,1:(N+k)]
        D = zeros(Float64, N+k+1, N+k)
        x = zeros(Float64, N+k)
        C = Matrix{Float64}(I, (N+k,N+k))
        # Full QR forced indices (logarithmic)
        @assert (0.0 <= pct_full_qr <= 100.0)
        fullQR_forced = floor(Int, m * pct_full_qr / 100)
        alpha = m ^ (1 / (fullQR_forced+1))
        full_forced_inds = zeros(Int, fullQR_forced)
        if fullQR_forced > 0
            full_forced_inds[1] = m - floor(Int, alpha) + 1
            for i in 2:fullQR_forced
                val = min(full_forced_inds[i-1]-1, m - floor(Int, alpha^i) + 1)
                if val < 1
                    full_forced_inds = full_forced_inds[1:i-1]
                    break
                end
                full_forced_inds[i] = val
            end
        end
        return new(V, Q, D, x, C, inds, ct, m, N, k, full_Q, full_forced_inds, SM_tol)
    end
end

function get_inds(kd::CholeskyDowndater)
    return kd.inds
end

function get_kernel_vectors(kd::CholeskyDowndater)
    N = kd.N; k = kd.k
    if kd.full_Q
        inds = kd.inds
        startidx = N+1
        kvecs = eachcol(view(kd.Q, inds, startidx:(N+k)))
        return kvecs
    else
        kvecs = eachcol(view(kd.Q, kd.inds, 1:(N+k))  * view(kd.C, 1:(N+k), (N+1):(N+k)))
        return kvecs
    end
end

function downdate!(kd::CholeskyDowndater, idx::Int)
    filter!(!=(idx), kd.inds)
    if kd.ct == kd.m
        return # No need to downdate anymore
    end
    kd.ct += 1
    
    x = kd.x; ct = kd.ct; N = kd.N; k = kd.k
    if kd.full_Q
        x .= view(kd.Q, idx, 1:(N+k))
    else
        x .= transpose(kd.C) * view(kd.Q, idx, 1:(N+k))
    end
    
    perform_fullQR = (length(kd.full_forced_inds) > 0 && ct == kd.full_forced_inds[end])

    if kd.k > (kd.m - (kd.ct - 1))
        # Reduce number of kernel vectors formed
        kd.k = (kd.m - (kd.ct - 1))
        k = kd.k
        perform_fullQR = true
        # Resize matrices
        kd.Q = view(kd.Q, :, 1:(N+k))
        kd.D = zeros(Float64, N+k+1, N+k)
        kd.x = zeros(Float64, N+k)
        kd.C = Matrix{Float64}(I, (N+k,N+k))
    else
        SM_sqrt_denom = 1 - x'x
        if (SM_sqrt_denom <= kd.SM_tol)
            perform_fullQR = true
        end
    end

    if perform_fullQR
        # Perform full QR reset
        Qnew, _ = qr(view(kd.V, kd.inds, 1:N))
        kd.Q[kd.inds, 1:(N+k)] .= Qnew[eachindex(kd.inds),1:(N+k)]
        kd.C .= Matrix(I, (N+k,N+k))
        if (length(kd.full_forced_inds) > 0 && ct == kd.full_forced_inds[end])
            pop!(kd.full_forced_inds)
        end
        return
    else
        # Cholesky downdate via Givens
        x .= (1 / sqrt(SM_sqrt_denom)) .* x
        D = kd.D
        D[1:(N+k),1:(N+k)] .= Matrix(I, (N+k,N+k))
        D[N+k+1,1:(N+k)] .= x
        for i in (N+k):-1:1
            G, r = givens(1.0, D[N+k+1,i], i, N+k+1)
            lmul!(G, D)
        end
        LinvTnew = transpose(view(D, 1:(N+k), 1:(N+k)))
        if kd.full_Q
            kd.Q[kd.inds,1:(N+k)] .= view(kd.Q, kd.inds, 1:(N+k)) * LinvTnew
        else
            kd.C .= kd.C * LinvTnew
        end
    end
end

"""
`FullQRUpDowndater`

A mutable struct to hold the QR decomposition of the matrix `V[inds,:]` 
for generating vectors in the kernel of its transpose. Only acts on 
`N+k` indices at a time. When downdate is called, it removes that index, 
and adds one of the remaining index, calling a new full QR factorization to 
complete the down and update. Takes `O((N+k)³)` flops.

Form with `FullQRUpDowndater(V[; ind_order=1:(size(V,1)), k=1])`.
`ind_order` is the order in which the indices are added. `k` is the 
(maximum) number of kernel vectors returned each time `get_kernel_vectors` is called. 
"""
mutable struct FullQRUpDowndater <: KernelDowndater
    V::AbstractMatrix
    Q::Union{AbstractMatrix,QRCompactWYQ}
    ind_order::AbstractVector
    inds::AbstractVector
    ct::Int
    m::Int
    N::Int
    k::Int
    function FullQRUpDowndater(V::AbstractMatrix; ind_order=randperm(size(V,1)), k=1)
        M, N = size(V)
        m = M - N
        if k > m
            k = m
        end
        ct = 1
        # Allocate arrays
        ind_order = collect(ind_order)
        inds = ind_order[1:(N+k)]
        Q,_ = qr(view(V, inds, 1:N))
        return new(V, Q, ind_order, inds, ct, m, N, k)
    end
end

function get_inds(kd::FullQRUpDowndater)
    return kd.inds
end

function get_kernel_vectors(kd::FullQRUpDowndater)
    N = kd.N; k = kd.k
    kvecs = eachcol(view(kd.Q, :, (N+1):(N+k)))
    return kvecs
end

function downdate!(kd::FullQRUpDowndater, idx::Int)
    pruneidx = findfirst(==(idx), kd.inds)
    if kd.ct == kd.m
        deleteat!(kd.inds, pruneidx)
        return # No need to downdate anymore
    end
    
    pruneidx = findfirst(==(idx), kd.inds)

    N = kd.N; k = kd.k; ct = kd.ct; m = kd.m


    if k == (m - ct + 1) # k + N + ct == M + 1, no more vectors to choose from
        kd.k -= 1
        deleteat!(kd.inds, pruneidx)
    else
        kd.inds[pruneidx] = kd.ind_order[k + N + ct]
    end

    # Perform full QR down and update
    Qnew, _ = qr(view(kd.V, kd.inds, 1:N))
    kd.Q = Qnew
    kd.ct += 1
end

"""
`GivensUpDowndater`

A mutable struct to hold the QR decomposition of the matrix `V[inds,:]` 
for generating vectors in the kernel of its transpose. Only acts on 
`N+k` indices at a time. When downdate is called, it removes that index, 
and adds one of the remaining index, using Givens rotations to 
complete the down and update. Takes `O((N+k)²)` flops.

Form with `GivensUpDowndater(V[; ind_order=1:(size(V,1)), k=1])`.
`ind_order` is the order in which the indices are added. `k` is the 
(maximum) number of kernel vectors returned each time `get_kernel_vectors` is called. 
"""
mutable struct GivensUpDowndater <: KernelDowndater
    V::AbstractMatrix
    Q::AbstractMatrix
    R::AbstractMatrix
    ind_order::AbstractVector
    inds::AbstractVector
    ct::Int
    m::Int
    N::Int
    k::Int
    function GivensUpDowndater(V::AbstractMatrix; ind_order=randperm(size(V,1)), k=1)
        M, N = size(V)
        m = M - N
        if k > m
            k = m
        end
        ct = 1
        # Allocate arrays
        ind_order = collect(ind_order)
        inds = ind_order[1:(N+k)]
        Q,R = qr(view(V, inds, 1:N))
        Q = Q[1:(N+k),1:(N+k)]
        R = vcat(R, zeros(k, N))
        return new(V, Q, R, ind_order, inds, ct, m, N, k)
    end
end

function get_inds(kd::GivensUpDowndater)
    return kd.inds
end

function get_kernel_vectors(kd::GivensUpDowndater)
    inds = kd.inds; N = kd.N; k = kd.k
    startidx = max(length(inds)-k+1, N+1)
    kvecs = eachcol(view(kd.Q, 1:(N+k), (N+1):(N+k)))
    return kvecs
end

function downdate!(kd::GivensUpDowndater, idx::Int)
    pruneidx = findfirst(==(idx), kd.inds)
    if kd.ct == kd.m
        deleteat!(kd.inds, pruneidx)
        return # No need to downdate anymore
    end

    N = kd.N; k = kd.k; ct = kd.ct; m = kd.m
    inds = kd.inds

    # Givens downdate
    q = view(kd.Q, pruneidx, 1:(N+k))
    r = q[N+k]
    for i in (N+k):-1:(pruneidx+1)
        G, r = givens(q[i-1], r, i-1, i)
        rmul!(kd.Q, G')
        lmul!(G, kd.R)
    end
    for i in (pruneidx-1):-1:1
        G, r = givens(r, q[i], pruneidx, i)
        rmul!(kd.Q, G')
        lmul!(G, kd.R)
    end

    if k == (m - ct + 1)
        # Reduce number of kernel vectors formed
        if kd.k > 1
            kd.k -= 1
            deleteat!(inds, pruneidx)
            newinds = [1:pruneidx-1 ; pruneidx+1:(N+k)]
            kd.Q = view(kd.Q, newinds, newinds)
        end
    else
        inds[pruneidx] = kd.ind_order[k + N + ct]
        # Givens update
        newrow = view(kd.V, inds[pruneidx], 1:N)
        kd.Q[pruneidx, pruneidx] = 1.0
        kd.R[pruneidx, :] .= newrow
        for i in 1:min(pruneidx-1, N)
            G, r = givens(kd.R[i,i], kd.R[pruneidx,i], i, pruneidx)
            rmul!(kd.Q, G')
            lmul!(G, kd.R)
        end
        for i in pruneidx:N
            G, r = givens(kd.R[i,i], kd.R[i+1,i], i, i+1)
            rmul!(kd.Q, G')
            lmul!(G, kd.R)
        end
    end
    kd.ct += 1
end