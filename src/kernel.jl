abstract type KernelDowndater end

# Required methods for KernelDowndater objects
function get_inds(kd::KernelDowndater)
    error("Must implement inds for KernelDowndater")
end
function get_kernel_vectors(kd::KernelDowndater)
    error("Must implement get_kernel_vectors for KernelDowndater")
end
function downdate!(kd::KernelDowndater, idx::Int) 
    error("Must implement downdate! for KernelDowndater")
end

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

function get_inds(fd::FullQRDowndater)
    return fd.inds
end

function get_kernel_vectors(fd::FullQRDowndater)
    startidx = max(size(fd.Q, 2)-fd.k+1, fd.N+1)
    kvecs = eachcol(fd.Q[:, startidx:size(fd.Q, 2)])
    return kvecs
end

function downdate!(fd::FullQRDowndater, idx::Int)
    if size(fd.Q,1) == (size(fd.V, 2) + 1)
        # Don't need to update anymore
        return
    end
    filter!(!=(idx), fd.inds)
    newQ, _ = qr(view(fd.V, fd.inds, 1:fd.N))
    fd.Q = newQ
end

mutable struct GivensQRDowndater <: KernelDowndater
    V::AbstractMatrix
    Q::Union{AbstractMatrix,QRCompactWYQ}
    inds::AbstractVector
    N::Int
    k::Int
    function GivensQRDowndater(V::AbstractMatrix; k=1)
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

function get_inds(gd::GivensQRDowndater)
    return gd.inds
end

function get_kernel_vectors(gd::GivensQRDowndater)
    inds = gd.inds
    startidx = max(length(inds)-gd.k+1, gd.N+1)
    kvecs = eachcol(view(gd.Q, inds, view(inds, startidx:length(inds))))
    return kvecs
end

function downdate!(gd::GivensQRDowndater, idx::Int)
    inds = gd.inds
    if length(inds) == (size(gd.V, 2) + 1)
        # Don't need to update anymore
        return
    end
    q = view(gd.Q, idx, inds)
    r = q[end]
    mididx = findfirst(==(idx), gd.inds)
    for i in length(q):-1:(mididx+1)
        G, r = givens(q[i-1], r, i-1, i)
        rmul!(view(gd.Q,inds,inds), G')
    end
    for i in (mididx-1):-1:1
        G, r = givens(r, q[i], mididx, i)
        rmul!(view(gd.Q,inds,inds), G')
    end
    deleteat!(inds, mididx)
end

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
    function CholeskyDowndater(V::AbstractMatrix; k=1, pct_full_qr=5.0, SM_tol=1e-6, full_Q=false)
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

function get_inds(cd::CholeskyDowndater)
    return cd.inds
end

function get_kernel_vectors(cd::CholeskyDowndater)
    N = cd.N; k = cd.k
    if cd.full_Q
        inds = cd.inds
        startidx = N+1
        kvecs = eachcol(view(cd.Q, inds, startidx:(N+k)))
        return kvecs
    else
        kvecs = eachcol(view(cd.Q, cd.inds, 1:(N+k))  * view(cd.C, 1:(N+k), (N+1):(N+k)))
        return kvecs
    end
end

function downdate!(cd::CholeskyDowndater, idx::Int)
    if cd.ct == cd.m
        return # No need to downdate anymore
    end
    cd.ct += 1
    
    x = cd.x; ct = cd.ct; N = cd.N; k = cd.k
    if cd.full_Q
        x .= view(cd.Q, idx, 1:(N+k))
    else
        x .= transpose(cd.C) * view(cd.Q, idx, 1:(N+k))
    end
    filter!(!=(idx), cd.inds)
    
    perform_fullQR = (length(cd.full_forced_inds) > 0 && ct == cd.full_forced_inds[end])

    if cd.k > (cd.m - (cd.ct - 1))
        # Reduce number of kernel vectors formed
        cd.k = (cd.m - (cd.ct - 1))
        k = cd.k
        perform_fullQR = true
        # Resize matrices
        cd.Q = view(cd.Q, :, 1:(N+k))
        cd.D = zeros(Float64, N+k+1, N+k)
        cd.x = zeros(Float64, N+k)
        cd.C = Matrix{Float64}(I, (N+k,N+k))
    else
        SM_sqrt_denom = 1 - x'x
        if (SM_sqrt_denom <= cd.SM_tol)
            perform_fullQR = true
        end
    end

    if perform_fullQR
        # Perform full QR reset
        Qnew, _ = qr(view(cd.V, cd.inds, 1:N))
        cd.Q[cd.inds, 1:(N+k)] .= Qnew[eachindex(cd.inds),1:(N+k)]
        cd.C .= Matrix(I, (N+k,N+k))
        if (length(cd.full_forced_inds) > 0 && ct == cd.full_forced_inds[end])
            pop!(cd.full_forced_inds)
        end
        return
    else
        # Cholesky downdate via Givens
        x .= (1 / sqrt(SM_sqrt_denom)) .* x
        D = cd.D
        D[1:(N+k),1:(N+k)] .= Matrix(I, (N+k,N+k))
        D[N+k+1,1:(N+k)] .= x
        for i in (N+k):-1:1
            G, r = givens(1.0, D[N+k+1,i], i, N+k+1)
            lmul!(G, D)
        end
        LinvTnew = transpose(view(D, 1:(N+k), 1:(N+k)))
        if cd.full_Q
            cd.Q[cd.inds,1:(N+k)] .= view(cd.Q, cd.inds, 1:(N+k)) * LinvTnew
        else
            cd.C .= cd.C * LinvTnew
        end
    end
end

mutable struct CholeskyUpDowndater <: KernelDowndater
    V::AbstractMatrix
    Q::AbstractMatrix
    R::AbstractMatrix
    kvecs::AbstractVector
    D::AbstractMatrix
    x::AbstractVector
    ind_order::AbstractVector
    inds::AbstractVector
    ct::Int
    m::Int
    N::Int
    k::Int
    full_forced_inds::AbstractVector{<:Int}
    SM_tol::Float64
    function CholeskyUpDowndater(V::AbstractMatrix; ind_order=1:(size(V,1)), k=1, pct_full_qr=50.0, SM_tol=1e-6)
        M, N = size(V)
        m = M - N
        if k > m
            k = m
        end
        ct = 1
        # Allocate arrays
        ind_order = collect(ind_order)
        inds = ind_order[1:(N+k+1)]
        Q,R = qr(view(V, inds, 1:N))
        Q = Q[1:(N+k+1),1:(N+k)]
        R = vcat(R, zeros(k, N))
        D = zeros(Float64, N+k+1, N+k)
        kvecs = [zeros(M) for _ in 1:k]
        x = zeros(Float64, N+k)
        # Full QR forced indices (uniform)
        @assert (0.0 <= pct_full_qr <= 100.0)
        fullQR_forced = floor(Int, m * pct_full_qr / 100)
        # Perhaps change this
        if fullQR_forced == 0
            full_forced_inds = Int[]
        elseif fullQR_forced == 1
            full_forced_inds = [floor(Int, (m+1)/2)]
        else
            full_forced_inds = unique([floor(Int, i) for i in range(m, 1, length=fullQR_forced+1)[2:end]])
        end
        return new(V, Q, R, kvecs, D, x, ind_order, inds, ct, m, N, k, full_forced_inds, SM_tol)
    end
end

function get_inds(cud::CholeskyUpDowndater)
    return cud.inds
end

function get_kernel_vectors(cud::CholeskyUpDowndater)
    N = cud.N; k = cud.k
    kvecs = view(cud.Q, :, (N+1):(N+k))
    for i in 1:cud.k
        cud.kvecs[i] .= 0.0
        for j in eachindex(cud.inds)
            inner_idx = cud.inds[j]
            cud.kvecs[i][inner_idx] = kvecs[j, i]
        end
    end
    kvecs = view(cud.kvecs, 1:cud.k)
    return kvecs
end

function downdate!(cud::CholeskyUpDowndater, idx::Int)
    if cud.ct == cud.m
        return # No need to downdate anymore
    end
    
    pruneidx = findfirst(==(idx), cud.inds)

    x = cud.x; N = cud.N; k = cud.k; ct = cud.ct; m = cud.m
    x .= view(cud.Q, pruneidx, 1:(N+k))
    
    perform_fullQR = (length(cud.full_forced_inds) > 0 && ct == cud.full_forced_inds[end])

    if k == (m - ct) # k + N + ct + 1 == M + 1, no more vectors to choose from
        if k == 1
            # Perform final QR, keep k at 1
            perform_fullQR = true
            # Resize matrices
            cud.Q = view(cud.Q, 1:(N+k), 1:(N+k))
            cud.R = view(cud.R, 1:(N+k), 1:N)
            cud.D = view(cud.D, 1:(N+k+1), (1:N+k))
            cud.x = view(cud.x, 1:(N+k))
            deleteat!(cud.inds, pruneidx)
        else
            # Reduce number of kernel vectors formed
            cud.k = m - ct - 1
            k = cud.k
            perform_fullQR = true
            # Resize matrices
            cud.Q = view(cud.Q, 1:(N+k+1), 1:(N+k))
            cud.R = view(cud.R, 1:(N+k), 1:N)
            cud.D = view(cud.D, 1:(N+k+1), (1:N+k))
            cud.x = view(cud.x, 1:(N+k))
            deleteat!(cud.inds, pruneidx)
        end
    else
        cud.inds[pruneidx] = cud.ind_order[ct+N+k+1]
        SM_sqrt_denom = 1 - x'x
        if (SM_sqrt_denom <= cud.SM_tol)
            perform_fullQR = true
        end
    end

    inds = cud.inds

    if perform_fullQR
        # Perform full QR down and update
        Qnew, Rnew = qr(view(cud.V, inds, 1:N))
        cud.Q[eachindex(inds), 1:(N+k)] .= Qnew[eachindex(inds),1:(N+k)]
        cud.R[1:N, 1:N] .= Rnew
        if (length(cud.full_forced_inds) > 0 && ct == cud.full_forced_inds[end])
            pop!(cud.full_forced_inds)
        end
    else
        # Cholesky downdate via Givens
        x .= (1 / sqrt(SM_sqrt_denom)) .* x
        D = cud.D
        D[1:(N+k),1:(N+k)] .= Matrix(I, (N+k,N+k))
        D[N+k+1,1:(N+k)] .= x
        for i in (N+k):-1:1
            G, r = givens(1.0, D[N+k+1,i], i, N+k+1)
            lmul!(G, D)
        end
        LinvTnew = UpperTriangular(transpose(view(D, 1:(N+k), 1:(N+k))))
        rmul!(view(cud.Q,eachindex(inds), 1:(N+k)), LinvTnew)
        rdiv!(transpose(cud.R), transpose(LinvTnew))# Instead of ldiv!(LinvTnew, cud.R), saves significant run-dispatch time
        # Cholesky update 
        newrow = view(cud.V, inds[pruneidx], 1:N)
        x[1:N] .= transpose(UpperTriangular(view(cud.R, 1:N, 1:N))) \ newrow
        x[N+1:N+k] .= 0.0
        cud.Q[pruneidx, 1:(N+k)] .= x
        D[1:(N+k),1:(N+k)] .= Matrix(I, (N+k,N+k))
        D[N+k+1,1:(N+k)] .= x
        for i in 1:N
            G, r = givens(1.0, D[N+k+1,i], i, N+k+1)
            lmul!(G, D)
        end
        LTnew = UpperTriangular(view(D, 1:(N+k), 1:(N+k)))
        rdiv!(view(cud.Q,eachindex(inds), 1:(N+k)), LTnew)
        lmul!(LTnew, cud.R)
    end
    cud.ct += 1
end