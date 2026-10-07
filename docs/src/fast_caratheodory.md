# Fast Carathéodory Pruning

`fast_caratheodory` accelerates pruning when the number of rows ``M`` is much
larger than the number of moments ``N``. It partitions the rows into small,
contiguous groups, replaces each group with its weighted moment contribution,
and prunes the group representatives. The rows in surviving groups are then
compacted and the procedure repeats. This avoids applying the base pruning
method directly to all ``M`` rows at once.

!!! warning "Requires in-memory arrays"
    `fast_caratheodory` does **not** support `OnDemandMatrix` or
    `OnDemandVector`. It allocates full intermediate matrix and weight buffers
    and compacts rows in place, so use it only when both `V` and `w` can be
    fully stored in memory. For on-demand inputs, use `caratheodory_pruning`.

```@docs
fast_caratheodory
```

## Example

The pruned rule preserves the weighted moments of the original rule.

```@example
using CaratheodoryPruning
using LinearAlgebra
using Random

Random.seed!(1)
M, N = 1_000, 10
V = rand(M, N)
w = rand(M)
w_pruned, inds = fast_caratheodory(V, w)

norm(transpose(V) * w - transpose(V) * w_pruned)
```

## Timing

Verification of runtime improvement: 

```@example
using CaratheodoryPruning
using Random

Random.seed!(1)
M, N = 1_000, 10
V = rand(M, N)
w = rand(M)
# Run each once to avoid compilation time
caratheodory_pruning(V, w)
fast_caratheodory(V, w)

t1 = @elapsed caratheodory_pruning(V, w)
t2 = @elapsed fast_caratheodory(V, w)

"Standard method: $(1000*t1)ms, fast method: $(1000*t2)ms"
```

## Method and reference

The method is inspired by the fast Carathéodory-set construction of Alaa
Maalouf, Ibrahim Jubran, and Dan Feldman, [*Fast and Accurate Least-Mean-Squares
Solvers*](https://proceedings.neurips.cc/paper_files/paper/2019/file/475fbefa9ebfba9233364533aafd02a3-Paper.pdf),
NeurIPS 2019. The paper combines balanced partitions, group representatives,
and repeated Carathéodory reductions; it identifies a group count proportional
to ``eN`` as a useful choice for a particular underlying solver. Accordingly,
`fast_caratheodory` defaults to `frac = MathConstants.e`.

This implementation adapts that idea to this package's weighted-row pruning
interface. In particular, it works with the general weight types supported by
`caratheodory_pruning`; it is not a verbatim implementation of the paper's
nonnegative-real Carathéodory-set algorithm.
