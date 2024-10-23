"""
`OnDemandMatrix{T} <: AbstractMatrix{T} where T`

Matrix subtype for which only a subset of the columns
or rows (determined by if `cols=true`) need to be stored
at a time in case storing the full matrix is too expensive.

To form an `n×m` "identity" `OnDemandMatrix` which is stored
column-wise, call

`M = OnDemandMatrix(n, m, j -> [1.0*(i==j) for i in 1:n])`.

To store the matrix row-wise, call

`M = OnDemandMatrix(n, m, i -> [1.0*(i==j) for j in 1:m], by=:rows)`.

Will only call the provided column or row function when trying
to access in that column or row.

To "forget" or erase a given column or row, indexed by `i`, call

`forget!(M, i)`.

Useful for use with `caratheodory_pruning` with the kernel option `:GivensUpDown`
where only a subset of rows/columns are required at a time.
"""
struct OnDemandMatrix{T} <: AbstractMatrix{T}
    n::Int
    m::Int
    vecs::Dict{Int,AbstractVector{T}}
    vecfun::Function
    cols::Bool
end

"""
`OnDemandMatrix(n::Int, m::Int, vecfun::Function; by=:cols, T=Float64)`

Forms an `n×m` `OnDemandMatrix` whose columns (or rows if `by==:rows`) are
formed when needed by `vecfun(i::Int)`. Default type is `T=Float64`.
"""
function OnDemandMatrix(n::Int, m::Int, vecfun::Function; by=:cols, T=Float64)
    @assert (by in (:rows, :cols)) "by argument must be either :cols or :rows"
    vecs = Dict{Int,AbstractVector{T}}()
    return OnDemandMatrix{T}(n, m, vecs, vecfun, by==:cols)
end

function Base.size(M::OnDemandMatrix)
    return (M.n,M.m)
end

function Base.show(io::Core.IO, mime::MIME"text/plain", M::OnDemandMatrix{T}) where T
    stored = length(M.vecs)
    print(io, "$(size(M,1))x$(size(M,2)) OnDemandMatrix{$T} with $(stored) stored $((M.cols ? "columns" : "rows"))")
end

# TODO: Is this the correct way to separate @show from print calls?
function Base.show(io::Core.IO, M::OnDemandMatrix{T}) where T
    print(io, "OnDemandMatrix{$T}(")
    print(io, "vecs=$(M.vecs),")
    print(io, "vecfun=$(M.vecfun),")
    print(io, "cols=$(M.cols))")
end

function Base.getindex(M::OnDemandMatrix, idx::Vararg{Int,2})
    i,j = idx
    if M.cols
        if !(j in keys(M.vecs))
            push!(M.vecs, j => M.vecfun(j))
        end
        return M.vecs[j][i]
    else
        if !(i in keys(M.vecs))
            push!(M.vecs, i => M.vecfun(i))
        end
        return M.vecs[i][j]
    end
end

"""
`forget!(M::OnDemandMatrix, i::Int)`

Erases from memory the given column or row
(determined by `M.cols`). If the given column
or row is not stored, will do nothing.
"""
function forget!(M::OnDemandMatrix, i::Int)
    pop!(M.vecs, i, nothing)
    M
end

function Base.transpose(M::OnDemandMatrix{T}) where T
    return OnDemandMatrix{T}(size(M,2), size(M,1), M.vecs, M.vecfun, !M.cols)
end

function Base.view(M::OnDemandMatrix, i::Int, js::Union{<:AbstractVector{Int},Colon})
    if (!M.cols)
        if !(i in keys(M.vecs))
            push!(M.vecs, i => M.vecfun(i))
        end
        return view(M.vecs[i], js)
    else
        return view(M, i:i, js)
    end
end

function Base.view(M::OnDemandMatrix, is::Union{<:AbstractVector{Int},Colon}, j::Int)
    if (M.cols)
        if !(j in keys(M.vecs))
            push!(M.vecs, j => M.vecfun(j))
        end
        return view(M.vecs[j], is)
    else
        return view(M, is, j:j)
    end
end