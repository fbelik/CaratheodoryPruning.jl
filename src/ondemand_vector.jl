"""
`OnDemandVector{T} <: AbstractVector{T} where T`

Vector subtype for which only a subset of the elements
need to be stored at a time in case storing the full 
vector is too costly.

To form a length `n` `OnDemandVector` with value `1` at 
index 1 and zero otherwise, call

`v = OnDemandVector(n, i -> 1.0 * (i == 1))`.

To "forget" or erase a given element, indexed by `i`, call

`forget!(v, i)`.

Useful for use with `caratheodory_pruning` with the kernel option `:GivensUpDown`
where only a subset of elements are required at a time.
"""
struct OnDemandVector{T} <: AbstractVector{T}
    n::Int
    elems::Dict{Int,T}
    elemfun::Function
end

"""
`OnDemandVector(n::Int, elemfun::Function; T=Float64)`

Forms an length `n` `OnDemandVector` who's elements are
determined by `elemfun(i)`. If an element is set to zero,
that element is removed from memory. Default type is `T=Float64`.
"""
function OnDemandVector(n::Int, elemfun::Function; T=Float64)
    elems = Dict{Int,T}()
    return OnDemandVector{T}(n, elems, elemfun)
end

function Base.size(M::OnDemandVector)
    return (M.n,)
end

function Base.show(io::Core.IO, mime::MIME"text/plain", M::OnDemandVector)
    stored = length(M.elems)
    print(io, "$(size(M,1))-element $(typeof(M)) with $(stored) stored elements")
end

# TODO: Is this the correct way to separate @show from print calls?
function Base.show(io::Core.IO, M::OnDemandVector)
    print(io, "$(typeof(M))(")
    print(io, "elems=$(M.elems),")
    print(io, "elemfun=$(M.elemfun))")
end

function Base.copy(M::OnDemandVector{T}) where T
    return OnDemandVector{T}(M.n, copy(M.elems), M.elemfun)
end

function Base.getindex(M::OnDemandVector{T}, idx::Vararg{Int,1}) where T
    i, = idx
    if !(i in keys(M.elems))
        val = M.elemfun(i)
        push!(M.elems, i => val)
    end
    return M.elems[i]
end

function Base.setindex!(M::OnDemandVector{T}, v::T, idx::Vararg{Int,1}) where T
    i, = idx
    if i in keys(M.elems)
        M.elems[i] = v
    else
        push!(M.elems, i => v)
    end
    return v
end

"""
`forget!(M::OnDemandVector, i::Int)`

Erases from memory the given element. 
If the given element is not stored, 
will do nothing.
"""
function forget!(M::OnDemandVector, i::Int)
    pop!(M.elems, i, nothing)
    M
end