"""
`VandermondeVector{T,TV<:AbstractVector{T},TP} <: AbstractVector{T}`

Vector subtype useful for storing a slice (row or column) of a Vandermonde 
matrix along with the point used to form the slice. The slice is stored in 
`vec` and the point is stored in `pt`. Behaves like a normal vector with 
contents `vec`.

Given `v = VandermondeVector(vec, pt)`, the point can be accessed with 
`getpt(v)` or `v.pt`.
"""
struct VandermondeVector{T,TV<:AbstractVector{T},TP} <: AbstractVector{T}
    vec::TV
    pt::TP
end

"""
`VandermondeVector(vec::TV, addl::TP) where TV<:AbstractVector{T} where T where TP`

Forms a `VandermondeVector` with contents `vec` and corresponding point
`pt`.
"""
function VandermondeVector(vec::TV, pt::TP=nothing) where TV<:AbstractVector{T} where T where TP
    return VandermondeVector{T,TV,TP}(vec,pt)
end

function Base.size(v::VandermondeVector)
    return size(v.vec)
end

function Base.getindex(v::VandermondeVector, idx::Vararg{Int,1})
    return getindex(v.vec, idx[1])
end

function Base.setindex!(v::VandermondeVector, val, idx::Vararg{Int,1})
    setindex!(v.vec, val, idx[1])
end

"""
`getpt(v::VandermondeVector)`

Returns the stored point, `v.pt`.
"""
function getpt(v::VandermondeVector)
    return v.pt
end

function Base.similar(v::VandermondeVector{T,TV,TP}) where T where TV where TP
    return VandermondeVector{T,TV,TP}(similar(v.vec), v.pt)
end

function Base.similar(v::VandermondeVector{T,TV,TP}, S::Type) where T where TV where TP
    vec = similar(v.vec, S)
    TVnew = typeof(vec)
    return VandermondeVector{S,TVnew,TP}(vec, v.pt)
end

Base.Broadcast.BroadcastStyle(::Type{<:VandermondeVector}) = Base.Broadcast.ArrayStyle{VandermondeVector}()

function Base.similar(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{VandermondeVector}}, S::Type)
    v = find_vv(bc)
    VandermondeVector(similar(v.vec, S), v.pt)
end

# For broadcasting purposes: https://docs.julialang.org/en/v1/manual/interfaces/#writing-binary-broadcasting-rules
find_vv(bc::Base.Broadcast.Broadcasted) = find_vv(bc.args)
find_vv(args::Tuple) = find_vv(find_vv(args[1]), Base.tail(args))
find_vv(x) = x
find_vv(::Tuple{}) = nothing
find_vv(v::VandermondeVector, rest) = v
find_vv(::Any, rest) = find_vv(rest)