# On Demand Matrices and Vectors

In the case that the matrix ``V`` is too large to store in memory, when using a `KernelDowndater` such as `GivensUpDowndater` which only requires access to a select number of rows of ``V`` at a time, we have implemented a matrix type `OnDemandMatrix` which stores only the required rows (or columns) at a time. The `GivensUpDowndater` or `FullQRUpDowndater` will automatically delete unneeded rows/columns throughout the pruning procedure. 

```@docs
OnDemandMatrix
```

Forming an `OnDemandMatrix` requires passing the full-size of the matrix, the function that generates new rows or columns, and whether it is to be stored column-wise or row-wise. This type works nicely with transposition.

To manually erase or forget a given row/column, call the `forget!` method.

```@docs
forget!
```

Similarly, if the larger dimension of ``V`` is so large that we wish to have complete memory independence from that dimension, we can store the weights in an `OnDemandVector` which only stores the required elements at a time. The method `caratheodory_pruning` will automatically delete unneeded elements throughout the pruning procedure. 

```@docs
OnDemandVector
```

The method `forget!` works the same for the `OnDemandVector`.

Related to on demand matrices is a `VandermondeVector` which is a thin wrapper for a standard vector which stores one additional piece of information such as the point used to generate the corresponding slice of the Vandermonde matrix.

```@docs
VandermondeVector
```

See the Monte Carlo example for use of this.