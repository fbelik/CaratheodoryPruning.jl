module CaratheodoryPruning

using LinearAlgebra: 
    qr, I, givens, lmul!, rmul!, ldiv!, rdiv!, QRCompactWYQ, norm, UpperTriangular, SingularException
using Random: randperm
using ProgressBars: ProgressBar, update

include("utils.jl")
include("ondemand_matrix.jl")
include("ondemand_vector.jl")
include("vandermonde_vec.jl")
include("kernel.jl")
include("pruning_weights.jl")
include("caratheodory.jl")

export OnDemandMatrix
export OnDemandVector
export forget!
export VandermondeVector
export getpt
export CholeskyDowndater
export FullQRDowndater
export GivensDowndater
export FullQRUpDowndater
export GivensUpDowndater
export caratheodory_pruning
export get_inds
export get_kernel_vectors
export prune_weights_first!
export prune_weights_minabs!
export downdate!

end
