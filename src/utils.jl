"""
`givens_qr_row_downdate!(Q, rowidx[, R=nothing])`

Performs an in-place Givens QR row downdate on the full Q-factor, 
and the R-factor if it is passed in. After the downdate, the new
Q factor is `Q[[1:rowidx-1 ; rowidx+1:end], [1:rowidx-1 ; rowidx+1:end]]`,
and the new R-factor is `R[[1:rowidx-1 ; rowidx+1:end], :]`.

Unlike with Givens updates, this method does not need the R-factor to
downdate the Q-factor.
"""
function givens_qr_row_downdate!(Q::AbstractMatrix, rowidx, R::Union{Nothing,AbstractMatrix}=nothing)
    n = size(Q,1)
    q = view(Q, rowidx, 1:n)
    r = q[n]
    for i in n:-1:(rowidx+1)
        G, r = givens(q[i-1], r, i-1, i)
        rmul!(Q, G')
        if !isnothing(R)
            lmul!(G, R)
        end
    end
    for i in (rowidx-1):-1:1
        G, r = givens(r, q[i], rowidx, i)
        rmul!(Q, G')
        if !isnothing(R)
            lmul!(G, R)
        end
    end
end

"""
`givens_qr_row_update!(Q, R, rowidx, newrow)`

Performs an in-place Givens QR row update on the full Q-factor, 
and the R-factor. `Q` is already expected to have an additional row 
and column allocated, so the old Q-factor is at 
`Q[[1:rowidx-1 ; rowidx+1:end], [1:rowidx-1 ; rowidx+1:end]]`, and the
new Q factor will fill the Q matrix. Similarly, the old R-factor is at 
`R[[1:rowidx-1 ; rowidx+1:end], :]`, and the new R factor will fill the 
R matrix. Afterwards, will have `(QR)[rowidx,:] == newrow`.
"""
function givens_qr_row_update!(Q::AbstractMatrix, R::AbstractMatrix, rowidx, newrow)
    n = size(R,2)
    Q[rowidx, rowidx] = 1.0
    R[rowidx, :] .= newrow
    for i in 1:min(rowidx-1, n)
        G, r = givens(R[i,i], R[rowidx,i], i, rowidx)
        rmul!(Q, G')
        lmul!(G, R)
    end
    for i in rowidx:n
        G, r = givens(R[i,i], R[i+1,i], i, i+1)
        rmul!(Q, G')
        lmul!(G, R)
    end
end