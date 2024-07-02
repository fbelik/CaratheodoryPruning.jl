module CaratheodoryPruning

using LinearAlgebra: 
    qr, I, givens, lmul!, rmul!, QRCompactWYQ, norm
using Random:
    randperm

include("kernel.jl")
include("pruning_weights.jl")
include("caratheodory.jl")

export CholeskyDowndater
export FullQRDowndater
export GivensQRDowndater
export CholeskyUpDowndater
export caratheodory_pruning
export get_kernel_vectors
export prune_weights_first!
export prune_weights_minabs!
export downdate!

end
