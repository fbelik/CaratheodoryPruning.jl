using CaratheodoryPruning

M = 100
N = 10

V = rand(M, N)
w = rand(M)

η = V'w

w_pruned, inds = caratheodory_pruning(V, w) # Cholesky Method
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:FullQR)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:GivensQR)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:Cholesky, full_Q=true)
# w_pruned, inds = caratheodory_pruning(V, w, kernel=:CholeskyUpDown)

error = sqrt(sum(x -> x^2, V[inds,:]'w_pruned[inds] .- η))

println("Error in quadrature moments is $error")