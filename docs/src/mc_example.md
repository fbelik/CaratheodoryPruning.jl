# Monte Carlo Quadrature Example

Consider the problem of integrating a circle with radius 1 centered at the origin in ``\mathbb{R}^2``, ``C``. One way of doing this is by Monte-Carlo sampling ``M`` points within the circle IID uniformly randomly, ``\{x_i\}_{i=1}^M`` and forming the following quadrature rule
```math
\int_C f(x) dx \approx \sum_{i=1}^M \frac{\pi}{M} f(x_i).
```
Suppose we wish to accurately integrate the set of bivariate polynomials of degree at most 4, given by the span of ``S = \{1, x, x^2, xy, y^2, \ldots, xy^3, y^4\}`` for which ``N = |S| = \binom{6}{2} = 15``. This means we can form a 15 point quadrature rule which maintains accuracy of the Monte-Carlo rule on ``S``. We can do this through pruning.

Suppose additionally that ``M`` is sufficiently large such that we do not wish to store the full set of ``M`` points in memory. This can be done with an `OnDemandMatrix` of `VandermondeVector`s and an `OnDemandVector`. See the following code for details.

```julia
using CaratheodoryPruning
M = 10000
p = 4 # Polynomial degree
d = 2 # in R²
N = binomial(p+d,d)


# Form the Vandermonde matrix OnDemand
vecfun(i) = begin
    # Form random point in C
    pt = 2 .* rand(d) .- 1
    while sum(pt .^ 2) > 1
        pt .= 2 .* rand(d) .- 1
    end
    # Evaluate point on basis
    vec = zeros(N)
    ct = 1
    for i in 0:p, j in 0:(p-i)
        vec[ct] = pt[1] ^ i * pt[2] ^ j
        ct += 1
    end
    # Store the vector and point in a Vandermonde vector
    return VandermondeVector(vec, pt)
end
# Store rows and pts OnDemand
V = OnDemandMatrix(M, N, vecfun, by=:rows, 
                   TV=VandermondeVector{Float64, Vector{Float64}, Vector{Float64}})

# Store the initial weights OnDemand
wfun(i) = π / M
w_in = OnDemandVector(M, wfun)

# Prune and compute reduced weights
w, inds, err = caratheodory_pruning(V, w_in)

# Function to evaluate reduced rule
quadrule(f) = begin
    res = 0.0
    for i in inds
        pt = V.vecs[i].pt
        wt = w[i]
        res += wt * f(pt)
    end
    return res
end
```