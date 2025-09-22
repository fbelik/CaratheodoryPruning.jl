# Pruning Options

Pruning is simply done by a method which takes in three arguments. 
They are the full vector of weights to update, `w`, a collection of indexable kernel vectors, `kvecs`, and the set of indicies from the `KernelDowndater`, `inds`.
The kernel vectors will each have the same length as `inds`, while `w` will have to be indexed by `inds`.
The pruning method will then choose a single or a linear combination of the kernel vectors to prune with, and update the weights vector, `w`.
The following method obtains the minimum perturbation along with the index zeroed out for a given weight vector and kernel vector.
```@docs
CaratheodoryPruning.get_min_alpha_k0
```

`CaratheodoryPruning.jl` comes with several built-in pruning options. They can be easily used by calling `caratheodory_pruning(V, w_in, pruning=PRUNING)`, replacing `PRUNING` with the appropriate method. 

### Prune first

Prunes using the first kernel vector in `kvecs`.

```@docs
prune_weights_first!
```
### Prune Minimum Absolute Value

Prunes according to the kernel vector in `kvecs` which results in the minimum absolute value multiple added.

```@docs
prune_weights_minabs!
```