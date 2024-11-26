using CaratheodoryPruning
using LinearAlgebra

M = 100
N = 10

V = rand(M, N)
w = rand(M)

η = V'w

w_pruned, inds = caratheodory_pruning(V, w) # GivensUpDown Method
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:FullQR)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:Givens)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:Cholesky)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:FullQRUpDown)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:GivensUpDown)

η_comp = V[inds,:]' * w_pruned[inds]

error = norm(η_comp .- η)

println("Error in quadrature moments is $error")