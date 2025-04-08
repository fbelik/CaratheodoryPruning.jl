using CaratheodoryPruning
using LinearAlgebra

M = 100
N = 10

V = rand(M, N)
w = rand(M)

η = V'w

w_pruned, inds = caratheodory_pruning(V, w) # GivensUpDown Method
# w_pruned, inds = caratheodory_pruning(V, w, kernel=FullQRDowndater)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:GivensDowndater)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:CholeskyDowndater)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:FullQRUpDowndater)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:GivensUpDowndater)

η_comp = V[inds,:]' * w_pruned[inds]

error = norm(η_comp .- η)

println("Error in quadrature moments is $error")