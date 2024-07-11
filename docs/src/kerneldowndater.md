# KernelDowndaters

A `KernelDowndater` is an abstract type used for computing kernel vectors of a subselection of the rows of the transpose of the matrix ``V``. 

Denote ``V_{S*}`` to be the matrix formed by subselecting the indices specified by ``S = \{s_1,\ldots,s_m\}``, and similarly, ``x_{S}`` to be the subselection of elements of ``x``.
This is typically done by forming QR-factorizations of ``V_{S*}``, and pulling a trailing column, ``q``, from the Q-factor, such that ``{V_{S*}}^T q = 0``.
The reason we are interested in forming kernel vectors is that then, we can update a set of weights without changing the moments.
```math
{V_{S*}}^T q = 0,\quad {V_{S*}}^T w_{S} = \eta \implies {V_{S*}}^T (w_{S} + \alpha q) = \eta,
```
where ``\alpha`` is a constant chosen to zero out one of the weights while keeping all others nonnegative. We then update the ``S``-indices of ``w`` as ``w_{S} = w_{S} + \alpha q``.

However, in Caratheodory pruning, at each step, ``S`` is typically only changed by removing (or sometimes adding) a few elements at a time.
Thus, it can be wasteful to fully recompute the QR decomposition at each step. 

`CaratheodoryPruning.jl` comes with several built-in Downdater options. They can be easily used by calling `caratheodory_pruning(V, w_in, kernel=:KERNEL)`, replacing `:KERNEL` with the appropriate symbol. 

To implement your own `KernelDowndater` type, create a struct that inherits from KernelDowndater, and implement the necessary methods.
```julia
struct MyKernelDowndater <: KernelDowndater
    V::AbstractMatrix
    # Other necessary components
end

function get_inds(kd::MyKernelDowndater)
    error("Still to be implemented")
end
function get_kernel_vectors(kd::MyKernelDowndater)
    error("Still to be implemented")
end
function downdate!(kd::MyKernelDowndater, idx::Int)
    error("Still to be implemented")
end
```

You can then use that downdater, along with say the `prune_weights_first!` pruning method, as follows.
```julia
kd = MyKernelDowndater(V, additional_args)
caratheodory_pruning(V, w_in, kd, prune_weights_first!)
```

Below are the available `KernelDowndater` options implemented in `CaratheodoryPruning.jl`.

### FullQRDowndater

Access with the kernel symbols `:FullQRDowndater` or `:FullQR`.

```@docs
FullQRDowndater
```

### GivensDowndater

Access with the kernel symbols `:GivensDowndater` or `:Givens`.

```@docs
GivensDowndater
```

### CholeskyDowndater

Access with the kernel symbols `:CholeskyDowndater` or `:Cholesky`.

```@docs
CholeskyDowndater
```

### FullQRUpDowndater

Access with the kernel symbols `:FullQRUpDowndater` or `:FullQRUpDown`.

```@docs
FullQRUpDowndater
```

### GivensUpDowndater

Access with the kernel symbols `:GivensUpDowndater` or `:GivensUpDown`.

```@docs
GivensUpDowndater
```