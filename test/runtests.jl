using CaratheodoryPruning
using Test
using Random: seed!
using LinearAlgebra: norm

@testset "CaratheodoryPruning.jl" begin
    seed!(1)
    V = rand(100,10)
    w = rand(size(V, 1))
    Vtw = V'w
    tol = 1e-12
    @testset "Kernel choice $kernel" for kernel in (:FullQR, :GivensQR, :Cholesky, :CholeskyUpDown)
        for pruning in (:first, :minabs)
            neww, inds = caratheodory_pruning(V, w, kernel=kernel, pruning=pruning, k=5)
            @test norm(V'neww .- Vtw) <= tol
        end
    end
end
