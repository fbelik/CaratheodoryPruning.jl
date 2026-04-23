using CaratheodoryPruning
using Test
using Random: seed!
using LinearAlgebra: norm, I

@testset "CaratheodoryPruning.jl" begin
    seed!(1)
    V = rand(100,10)
    w = rand(size(V, 1))
    Vtw = transpose(V) * w
    w_arbsn = randn(size(V, 1))
    Vtw_arbsn = transpose(V) * w_arbsn
    V_cplx = randn(ComplexF64, 100, 10)
    w_cplx = randn(ComplexF64, size(V_cplx, 1))
    Vtw_cplx = transpose(V_cplx) * w_cplx
    @inferred caratheodory_pruning(V, w)
    @inferred caratheodory_pruning(V, w_arbsn)
    @inferred caratheodory_pruning(V_cplx, w_cplx)
    tol = 1e-12
    @testset "Kernel choice $kernel" for kernel in (FullQRDowndater, GivensDowndater, CholeskyDowndater, FullQRUpDowndater, GivensUpDowndater)
        for pruning in (prune_weights_first!, prune_weights_minabs!)
            for caratheodory_correction in (true,false)
                for k in (1,5)
                    neww, inds = caratheodory_pruning(V, w, kernel=kernel, pruning=pruning, 
                        caratheodory_correction=caratheodory_correction, k=k)
                    @test norm(transpose(V) * neww .- Vtw) <= tol
                    # Arbitrary sign
                    neww, inds = caratheodory_pruning(V, w_arbsn, kernel=kernel, pruning=pruning, 
                        caratheodory_correction=caratheodory_correction, k=k)
                    @test norm(transpose(V) * neww .- Vtw_arbsn) <= tol
                    # Complex
                    neww, inds = caratheodory_pruning(V_cplx, w_cplx, kernel=kernel, pruning=pruning, 
                        caratheodory_correction=caratheodory_correction, k=k)
                    @test norm(transpose(V_cplx) * neww .- Vtw_cplx) <= tol
                end
            end
        end
    end
    @testset "VandermondeVector" begin
        vect = collect(1.0:1:5)
        pt = collect(1.0:1:1)
        v = VandermondeVector(vect, pt)
        @test v == vect
        @test v ./ v == ones(5)
        @test v.pt == pt
        @test getpt(v) == pt
    end
    @testset "OnDemandMatrix $selection-wise" for selection in (:cols,:rows)
        M = OnDemandMatrix(5, 5, i -> [1.0*(i==j) for j in 1:5], by=selection, T=Float64)
        M[2,2]
        @test (2 in keys(M.vecs))
        @test (M.vecs[2][2] == 1.0)
        @test (M.vecs[2][5] == 0.0)
        @test (length(keys(M.vecs)) == 1)
        # Complex case
        M = OnDemandMatrix(5, 5, i -> rand(ComplexF64, 5), by=selection, T=ComplexF64)
        @test norm(M .- transpose(transpose(M))) == 0
        @test norm(M .- adjoint.(adjoint(transpose(M)))) == 0
    end
    @testset "OnDemandVector" begin
        M = OnDemandVector(5, i -> 1.0*i, T=Float64)
        M[2]
        @test (2 in keys(M.elems))
        @test (M.elems[2] == 2.0)
        @test (length(keys(M.elems)) == 1)
    end
    @testset "OnDemandPruning" begin
        M = 100; N = 10
        V0 = rand(N,M)
        w0 = rand(M)
        V = OnDemandMatrix(N,M,i->V0[:,i],by=:cols)
        w_in = OnDemandVector(M,i->w0[i])
        seed!(1)
        @inferred caratheodory_pruning(V, w_in)
        seed!(1) # In case of randomness
        w1,inds1 = caratheodory_pruning(V0, w0)
        seed!(1) # In case of randomness
        w2,inds2 = caratheodory_pruning(V, w_in)
        @test inds1 == inds2
        @test norm(w1[inds1] .- w2[inds2]) <= 1e-6
        @test norm(V[:,inds2]*w2[inds2] .- V0[:,inds1]*w1[inds1]) <= 1e-6
        @test length(keys(w_in.elems)) == 0
        @test length(keys(w2.elems)) == N
        @test length(keys(V.vecs)) == N
        # Test if use row-storage instead
        V = OnDemandMatrix(N,M,i->V0[i,:],by=:rows)
        seed!(1) # In case of randomness
        w3,inds3 = caratheodory_pruning(V, w0)
        @test inds1 == inds3
        @test norm(w1[inds1] .- w3[inds3]) <= 1e-6
        @test norm(V[:,inds3]*w3[inds3] .- V0[:,inds1]*w1[inds1]) <= 1e-6
        @test length(keys(V.vecs)) == N
        @test all([length(V.vecs[i]) == M for i in 1:N])
        # Test complex valued
        @testset "Complex OnDemand Kernel Choice $kernel" for kernel in (FullQRDowndater, GivensDowndater, CholeskyDowndater, FullQRUpDowndater, GivensUpDowndater)
            M = 100; N = 10
            V0 = randn(ComplexF64,N,M)
            w0 = randn(ComplexF64,M)
            V = OnDemandMatrix(N,M,i->V0[:,i],by=:cols,T=ComplexF64)
            w_in = OnDemandVector(M,i->w0[i],T=ComplexF64)
            seed!(1) # In case of randomness
            w1,inds1 = caratheodory_pruning(V0, w0, kernel=kernel)
            seed!(1) # In case of randomness
            w2,inds2 = caratheodory_pruning(V, w_in, kernel=kernel)
            @test inds1 == inds2
            @test w1[inds1] ≈ w2[inds2]
        end
    end

    @testset "Extra Pruning" begin
        V = Matrix{Float64}(I, (10,10))
        V[end,end] = 1e-16
        V[end-1,end-1] = 1e-16
        w_in = ones(10)
        w, inds = caratheodory_pruning(V, w_in, extra_pruning=true)
        @test !(10 in inds)
        @test !(9 in inds)
        w, inds = caratheodory_pruning(V, w_in, extra_pruning=false)
        @test (10 in inds)
        @test (9 in inds)
        # Randomized low-rank positive sign
        for _ in 1:10
            V = zeros(5,5)
            w_in = rand(5)
            for _ in 1:4
                v = randn(5)
                V .+= v * v'
            end
            w, inds = caratheodory_pruning(V, w_in, extra_pruning=true, zero_tol=1e-14)
            @test transpose(V) * w_in ≈ transpose(V) * w
            @test length(inds) < 5
        end
        # Randomized low-rank arbitrary sign
        for _ in 1:10
            V = zeros(5,5)
            w_in = randn(5)
            for _ in 1:4
                v = randn(5)
                V .+= v * v'
            end
            w, inds = caratheodory_pruning(V, w_in, extra_pruning=true, zero_tol=1e-14)
            @test transpose(V) * w_in ≈ transpose(V) * w
            @test length(inds) < 5
        end
        # Randomized low-rank complex
        for _ in 1:10
            V = zeros(ComplexF64,5,5)
            w_in = randn(ComplexF64,5)
            for _ in 1:4
                v = randn(ComplexF64,5)
                V .+= v * v'
            end
            w, inds = caratheodory_pruning(V, w_in, extra_pruning=true, zero_tol=1e-14)
            @test transpose(V) * w_in ≈ transpose(V) * w
            @test length(inds) < 5
        end
    end

    @testset "Signed and Complex" begin
        M = 50; N = 10
        tol = 1e-12
        seed!(1)
        # Both signs
        V = rand(M, N)
        w_in = rand(M) .- 0.5
        w, inds = caratheodory_pruning(V, w_in)
        @test norm(V'w_in .- V[inds,:]'w[inds]) <= tol
        @test all(i -> (w_in[i] <= 0 && w[i] <= 0) || (w_in[i] >= 0 && w[i] >= 0), eachindex(w_in))
        # Negative
        w_in = -1 .* rand(M)
        w, inds = caratheodory_pruning(V, w_in)
        @test norm(V'w_in .- V[inds,:]'w[inds]) <= tol
        @test all(w .<= 0)
        # Complex
        V = rand(ComplexF64, M, N)
        w_in = rand(ComplexF64, M)
        w, inds = caratheodory_pruning(V, w_in)
        # Must use transpose instead of adjoint for moment matching
        @test norm(transpose(V)*w_in .- transpose(V[inds,:])*w[inds]) <= tol
        # Complex with transposed V
        V = rand(ComplexF64, N, M)
        w_in = rand(ComplexF64, M)
        w, inds = caratheodory_pruning(V, w_in)
        @test norm(V*w_in .- V[:,inds]*w[inds]) <= tol
    end
end
