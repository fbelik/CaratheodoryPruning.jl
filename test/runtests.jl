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
    @testset "Kernel choice $kernel" for kernel in (:FullQR, :Givens, :Cholesky, :FullQRUpDown, :GivensUpDown)
        for pruning in (:first, :minabs)
            for caratheodory_correction in (true,false)
                for k in (1,5)
                    neww, inds = caratheodory_pruning(V, w, kernel=kernel, pruning=pruning, 
                        caratheodory_correction=caratheodory_correction, k=k)
                    @test norm(V'neww .- Vtw) <= tol
                end
            end
        end
    end
    @testset "OnDemandMatrix $selection-wise" for selection in (:cols,:rows)
        M = OnDemandMatrix(5, 5, i -> [1.0*(i==j) for j in 1:5], by=selection, T=Float64)
        M[2,2]
        @test (2 in keys(M.vecs))
        @test (M.vecs[2][2] == 1.0)
        @test (M.vecs[2][5] == 0.0)
        @test (length(keys(M.vecs)) == 1)
    end
end
