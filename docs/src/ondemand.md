# On Demand Matrices

In the case that the matrix ``V`` is too large to store in memory, when using a `KernelDowndater` such as `GivensUpDowndater` which only requires access to a select number of rows of ``V`` at a time, we have implemented a matrix type `OnDemandMatrix` which stores only the required rows (or columns) at a time. The `GivensUpDowndater` or `FullQRUpDowndater` will automatically delete unneeded rows/columns throughout the pruning procedure. 

```@docs
OnDemandMatrix
```

Forming an `OnDemandMatrix` requires passing the full-size of the matrix, the function that generates new rows or columns, and whether it is to be stored column-wise or row-wise. This type works nicely with transposition.

To manually erase or forget a given row/column, call the `forget!` method.

```@docs
forget!
```