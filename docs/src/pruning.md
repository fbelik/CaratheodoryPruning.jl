# Pruning Options

Pruning is simply done by a method which takes in three arguments. 
They are the full vector of weights to update, `w`, a collection of indexable kernel vectors, `kvecs`, and the set of indicies from the `KernelDowndater`, `inds`.
The kernel vectors will each have the same length as `inds`, while `w` will have to be indexed by `inds`.
The pruning method will then choose a single or a linear combination of the kernel vectors to prune with, and update the weights vector, `w`.
For each kernel vector, there will typically be two choices of scalars to add to the weights to maintain nonnegativity.
These can be computed by the `get_alpha_k0s` method.
```@docs
CaratheodoryPruning.get_alpha_k0s
```

`CaratheodoryPruning.jl` comes with several built-in pruning options. They can be easily used by calling `caratheodory_pruning(V, w_in, pruning=:PRUNING)`, replacing `:PRUNING` with the appropriate symbol. 

### Prune first

Prunes using the first kernel vector in `kvecs`.
Access with the pruning symbol `:first`.

```@docs
prune_weights_first!
```
### Prune Minimum Absolute Value

Prunes according to the kernel vector in `kvecs` which results in the minimum absolute value multiple added.
Access with the pruning symbol `:minabs`.

```@docs
prune_weights_minabs!
```