var documenterSearchIndex = {"docs":
[{"location":"pruning.html#Pruning-Options","page":"Pruning","title":"Pruning Options","text":"","category":"section"},{"location":"pruning.html","page":"Pruning","title":"Pruning","text":"Pruning is simply done by a method which takes in three arguments.  They are the full vector of weights to update, w, a collection of indexable kernel vectors, kvecs, and the set of indicies from the KernelDowndater, inds. The kernel vectors will each have the same length as inds, while w will have to be indexed by inds. The pruning method will then choose a single or a linear combination of the kernel vectors to prune with, and update the weights vector, w. For each kernel vector, there will typically be two choices of scalars to add to the weights to maintain nonnegativity. These can be computed by the get_alpha_k0s method.","category":"page"},{"location":"pruning.html","page":"Pruning","title":"Pruning","text":"CaratheodoryPruning.get_alpha_k0s","category":"page"},{"location":"pruning.html#CaratheodoryPruning.get_alpha_k0s","page":"Pruning","title":"CaratheodoryPruning.get_alpha_k0s","text":"get_alpha_k0s(w, kvec, inds)\n\nHelper method that, given a vector of weights, w, a kernel vector kvec, and a vector of indices, inds, returns a 4-tuple, (alphan, k0n, alphap, k0p) used for pruning. alphan is the most negative multiple allowed such that w = w + alphan * kvec still has nonnegative entries, and equals zero at k0n. Similarly, alphap is the most positive multiple allowed such that w = w + alphap * kvec  still has nonnegative entries, and equals zero at k0p.\n\n\n\n\n\n","category":"function"},{"location":"pruning.html","page":"Pruning","title":"Pruning","text":"CaratheodoryPruning.jl comes with several built-in pruning options. They can be easily used by calling caratheodory_pruning(V, w_in, pruning=:PRUNING), replacing :PRUNING with the appropriate symbol. ","category":"page"},{"location":"pruning.html#Prune-first","page":"Pruning","title":"Prune first","text":"","category":"section"},{"location":"pruning.html","page":"Pruning","title":"Pruning","text":"Prunes using the first kernel vector in kvecs. Access with the pruning symbol :first.","category":"page"},{"location":"pruning.html","page":"Pruning","title":"Pruning","text":"prune_weights_first!","category":"page"},{"location":"pruning.html#CaratheodoryPruning.prune_weights_first!","page":"Pruning","title":"CaratheodoryPruning.prune_weights_first!","text":"prune_weights_first!(w, kvecs, inds)\n\nTakes in a vector of full-length weights, w, a vector of kernel vectors, kvecs, and a vector of indices, inds, to which the indices of the kernel vectors point in the weights.\n\nTakes the first kernel vector, and prunes with that, using the minimum absolute value multiple needed to zero one of the weights.\n\n\n\n\n\n","category":"function"},{"location":"pruning.html#Prune-Minimum-Absolute-Value","page":"Pruning","title":"Prune Minimum Absolute Value","text":"","category":"section"},{"location":"pruning.html","page":"Pruning","title":"Pruning","text":"Prunes according to the kernel vector in kvecs which results in the minimum absolute value multiple added. Access with the pruning symbol :minabs.","category":"page"},{"location":"pruning.html","page":"Pruning","title":"Pruning","text":"prune_weights_minabs!","category":"page"},{"location":"pruning.html#CaratheodoryPruning.prune_weights_minabs!","page":"Pruning","title":"CaratheodoryPruning.prune_weights_minabs!","text":"prune_weights_minabs!(w, kvecs, inds)\n\nTakes in a vector of full-length weights, w, a vector of kernel vectors, kvecs, and a vector of indices, inds, to which the indices of the kernel vectors point in the weights.\n\nLoops over all kernel vectors, and prunes with the  vector with the minimum absolute value multiple needed to zero one of the weights.\n\n\n\n\n\n","category":"function"},{"location":"kerneldowndater.html#KernelDowndaters","page":"Kernel Downdaters","title":"KernelDowndaters","text":"","category":"section"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"A KernelDowndater is an abstract type used for computing kernel vectors of a subselection of the rows of the transpose of the matrix V. ","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"Denote V_S* to be the matrix formed by subselecting the indices specified by S = s_1ldotss_m, and similarly, x_S to be the subselection of elements of x. This is typically done by forming QR-factorizations of V_S*, and pulling a trailing column, q, from the Q-factor, such that V_S*^T q = 0. The reason we are interested in forming kernel vectors is that then, we can update a set of weights without changing the moments.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"V_S*^T q = 0quad V_S*^T w_S = eta implies V_S*^T (w_S + alpha q) = eta","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"where alpha is a constant chosen to zero out one of the weights while keeping all others nonnegative. We then update the S-indices of w as w_S = w_S + alpha q.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"However, in Carathéodory pruning, at each step, S is typically only changed by removing (or sometimes adding) a few elements at a time. Thus, it can be wasteful to fully recompute the QR decomposition at each step. ","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"CaratheodoryPruning.jl comes with several built-in Downdater options. They can be easily used by calling caratheodory_pruning(V, w_in, kernel=:KERNEL), replacing :KERNEL with the appropriate symbol. ","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"To implement your own KernelDowndater type, create a struct that inherits from KernelDowndater, and implement the necessary methods.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"struct MyKernelDowndater <: KernelDowndater\n    V::AbstractMatrix\n    # Other necessary components\nend\n\nfunction get_inds(kd::MyKernelDowndater)\n    error(\"Still to be implemented\")\nend\nfunction get_kernel_vectors(kd::MyKernelDowndater)\n    error(\"Still to be implemented\")\nend\nfunction downdate!(kd::MyKernelDowndater, idx::Int)\n    error(\"Still to be implemented\")\nend","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"You can then use that downdater, along with say the prune_weights_first! pruning method, as follows.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"kd = MyKernelDowndater(V, additional_args)\ncaratheodory_pruning(V, w_in, kd, prune_weights_first!)","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"Below are the available KernelDowndater options implemented in CaratheodoryPruning.jl.","category":"page"},{"location":"kerneldowndater.html#FullQRDowndater","page":"Kernel Downdaters","title":"FullQRDowndater","text":"","category":"section"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"Access with the kernel symbols :FullQRDowndater or :FullQR.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"FullQRDowndater","category":"page"},{"location":"kerneldowndater.html#CaratheodoryPruning.FullQRDowndater","page":"Kernel Downdaters","title":"CaratheodoryPruning.FullQRDowndater","text":"FullQRDowndater <: KernelDowndater\n\nA mutable struct to hold the Q-factor of the QR decomposition of the matrix V[inds,:] for generating vectors in the kernel  of its transpose. Downdates by forming a new QR factor, which takes O(M N²) flops where V is an M x N matrix.\n\nForm with FullQRDowndater(V[; k=1]) where k is the (maximum) number of kernel vectors returned each time get_kernel_vectors is called.\n\n\n\n\n\n","category":"type"},{"location":"kerneldowndater.html#GivensDowndater","page":"Kernel Downdaters","title":"GivensDowndater","text":"","category":"section"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"Access with the kernel symbols :GivensDowndater or :Givens.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"GivensDowndater","category":"page"},{"location":"kerneldowndater.html#CaratheodoryPruning.GivensDowndater","page":"Kernel Downdaters","title":"CaratheodoryPruning.GivensDowndater","text":"GivensDowndater <: KernelDowndater\n\nA mutable struct to hold the Q-factor of the QR decomposition of the matrix V[inds,:] for generating vectors in the kernel  of its transpose. Downdates by applying Givens rotations to the old, full, Q-factor, which takes O(M²) flops where V is an M x N matrix.\n\nForm with GivensDowndater(V[; k=1]) where k is the (maximum) number of kernel vectors returned each time get_kernel_vectors is called.\n\n\n\n\n\n","category":"type"},{"location":"kerneldowndater.html#CholeskyDowndater","page":"Kernel Downdaters","title":"CholeskyDowndater","text":"","category":"section"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"Access with the kernel symbols :CholeskyDowndater or :Cholesky.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"CholeskyDowndater","category":"page"},{"location":"kerneldowndater.html#CaratheodoryPruning.CholeskyDowndater","page":"Kernel Downdaters","title":"CaratheodoryPruning.CholeskyDowndater","text":"CholeskyDowndater <: KernelDowndater\n\nA mutable struct to hold the Q-factor of the QR decomposition of the matrix V[inds,:] for generating vectors in the kernel  of its transpose. Downdates by reorthogonalizing the old Q-factor, with a row removed, by multiplication by the inverse transpose of its Cholesky factor, which takes O(N³ + MN) flops where V is an M x N matrix.\n\nForm with CholeskyDowndater(V[; k=1, pct_full_qr=10.0, SM_tol=1e-6, full_Q=false)]). k is the (maximum) number of kernel vectors returned each time  get_kernel_vectors is called. pct_full_qr is the percentage (between 0 and 100), of times, logarithmically spaced, that a full QR reset will be done to prevent accumulation of error. SM_tol is a tolerance on the denominator of the Sherman Morrison formula to prevent error from division close to zero. full_Q determines whether or not the full Q matrix is updated or just its Cholesky factor; if set to true, will take O(N³ + MN²) flops instead of O(N³ + MN).\n\nFrom testing, seems to have minimal error accumulation if pct_full_qr ≥ 10.0.\n\n\n\n\n\n","category":"type"},{"location":"kerneldowndater.html#FullQRUpDowndater","page":"Kernel Downdaters","title":"FullQRUpDowndater","text":"","category":"section"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"Access with the kernel symbols :FullQRUpDowndater or :FullQRUpDown.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"FullQRUpDowndater","category":"page"},{"location":"kerneldowndater.html#CaratheodoryPruning.FullQRUpDowndater","page":"Kernel Downdaters","title":"CaratheodoryPruning.FullQRUpDowndater","text":"FullQRUpDowndater <: KernelDowndater\n\nA mutable struct to hold the QR decomposition of the matrix V[inds,:]  for generating vectors in the kernel of its transpose. Only acts on  N+k indices at a time. When downdate is called, it removes that index,  and adds one of the remaining index, calling a new full QR factorization to  complete the down and update. Takes O((N+k)³) flops.\n\nForm with FullQRUpDowndater(V[; ind_order=randperm(size(V,1)), k=1]). ind_order is the order in which the indices are added. k is the  (maximum) number of kernel vectors returned each time get_kernel_vectors is called. \n\n\n\n\n\n","category":"type"},{"location":"kerneldowndater.html#GivensUpDowndater","page":"Kernel Downdaters","title":"GivensUpDowndater","text":"","category":"section"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"Access with the kernel symbols :GivensUpDowndater or :GivensUpDown.","category":"page"},{"location":"kerneldowndater.html","page":"Kernel Downdaters","title":"Kernel Downdaters","text":"GivensUpDowndater","category":"page"},{"location":"kerneldowndater.html#CaratheodoryPruning.GivensUpDowndater","page":"Kernel Downdaters","title":"CaratheodoryPruning.GivensUpDowndater","text":"GivensUpDowndater <: KernelDowndater\n\nA mutable struct to hold the QR decomposition of the matrix V[inds,:]  for generating vectors in the kernel of its transpose. Only acts on  N+k indices at a time. When downdate is called, it removes that index,  and adds one of the remaining index, using Givens rotations to  complete the down and update. Takes O((N+k)²) flops.\n\nForm with GivensUpDowndater(V[; ind_order=randperm(size(V,1)), k=1, pct_full_qr=2.0]). ind_order is the order in which the indices are added. k is the  (maximum) number of kernel vectors returned each time get_kernel_vectors is called.  pct_full_qr is the percent of times, linearly spaced, that full QR factorizations are performed to prevent error accumulation in Q.\n\n\n\n\n\n","category":"type"},{"location":"ondemand.html#On-Demand-Matrices-and-Vectors","page":"On Demand Matrices","title":"On Demand Matrices and Vectors","text":"","category":"section"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"In the case that the matrix V is too large to store in memory, when using a KernelDowndater such as GivensUpDowndater which only requires access to a select number of rows of V at a time, we have implemented a matrix type OnDemandMatrix which stores only the required rows (or columns) at a time. The GivensUpDowndater or FullQRUpDowndater will automatically delete unneeded rows/columns throughout the pruning procedure. ","category":"page"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"OnDemandMatrix","category":"page"},{"location":"ondemand.html#CaratheodoryPruning.OnDemandMatrix","page":"On Demand Matrices","title":"CaratheodoryPruning.OnDemandMatrix","text":"OnDemandMatrix{T} <: AbstractMatrix{T} where T\n\nMatrix subtype for which only a subset of the columns or rows (determined by if cols=true) need to be stored at a time in case storing the full matrix is too expensive.\n\nTo form an n×m \"identity\" OnDemandMatrix which is stored column-wise, call\n\nM = OnDemandMatrix(n, m, j -> [1.0*(i==j) for i in 1:n]).\n\nTo store the matrix row-wise, call\n\nM = OnDemandMatrix(n, m, i -> [1.0*(i==j) for j in 1:m], by=:rows).\n\nWill only call the provided column or row function when trying to access in that column or row.\n\nTo \"forget\" or erase a given column or row, indexed by i, call\n\nforget!(M, i).\n\nUseful for use with caratheodory_pruning with the kernel option :GivensUpDown where only a subset of rows/columns are required at a time.\n\n\n\n\n\n","category":"type"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"Forming an OnDemandMatrix requires passing the full-size of the matrix, the function that generates new rows or columns, and whether it is to be stored column-wise or row-wise. This type works nicely with transposition.","category":"page"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"To manually erase or forget a given row/column, call the forget! method.","category":"page"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"forget!","category":"page"},{"location":"ondemand.html#CaratheodoryPruning.forget!","page":"On Demand Matrices","title":"CaratheodoryPruning.forget!","text":"forget!(M::OnDemandMatrix, i::Int)\n\nErases from memory the given column or row (determined by M.cols). If the given column or row is not stored, will do nothing.\n\n\n\n\n\nforget!(M::OnDemandVector, i::Int)\n\nErases from memory the given element.  If the given element is not stored,  will do nothing.\n\n\n\n\n\n","category":"function"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"Similarly, if the larger dimension of V is so large that we wish to have complete memory independence from that dimension, we can store the weights in an OnDemandVector which only stores the required elements at a time. The method caratheodory_pruning will automatically delete unneeded elements throughout the pruning procedure. ","category":"page"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"OnDemandVector","category":"page"},{"location":"ondemand.html#CaratheodoryPruning.OnDemandVector","page":"On Demand Matrices","title":"CaratheodoryPruning.OnDemandVector","text":"OnDemandVector{T} <: AbstractVector{T} where T\n\nVector subtype for which only a subset of the elements need to be stored at a time in case storing the full  vector is too costly.\n\nTo form a length n OnDemandVector with value 1 at  index 1 and zero otherwise, call\n\nv = OnDemandVector(n, i -> 1.0 * (i == 1)).\n\nTo \"forget\" or erase a given element, indexed by i, call\n\nforget!(v, i).\n\nUseful for use with caratheodory_pruning with the kernel option :GivensUpDown where only a subset of elements are required at a time.\n\n\n\n\n\n","category":"type"},{"location":"ondemand.html","page":"On Demand Matrices","title":"On Demand Matrices","text":"The method forget! works the same for the OnDemandVector.","category":"page"},{"location":"index.html#CaratheodoryPruning.jl-Documentation","page":"Background","title":"CaratheodoryPruning.jl Documentation","text":"","category":"section"},{"location":"index.html","page":"Background","title":"Background","text":"Carathéodory's theorem is a theorem in convex geometry relating to a minimal number of points in R^N required to enclose some point. Suppose that you have a set of points P in mathbbR^N with P  N. Additionally, let mathbfx in textconv(P), the convex hull of P. This means that mathbfx can be written as a positive linear combination of the points in P where the coefficients sum to one. The theorem states that there exists a subset Qsubset P with Q=N+1 such that mathbfx in textconv(Q). In N=2 dimensions, this means that given some number of points that enclose mathbfx, we can always prune these down to three points, or a triangle, that enclose mathbfx.","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"(Image: )","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"This theorem also extends to conic hulls, where the coefficients of the linear combination need not add to one, they simply need to be nonnegative. In the conic case, with mathbfx in textcone(P), the conic hull of P, there exists a subset Qsubset P with Q=N such that mathbfx in textcone(Q).","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"We can write out the conic version of Carathéodory's theorem as follows. Denote the points in P as mathbfp_1 ldots mathbfp_M. Also define be the matrix","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"mathbfP = beginbmatrix \nvert  vert    vert\nmathbfp_1  mathbfp_2  cdots  mathbfp_M\nvert  vert   vert\nendbmatrix in mathbbR^N times M","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"The statement that mathbfxintextcone(P) implies that there exists a nonnegative vector of weights, mathbfw in mathbbR^M, such that","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"mathbfP mathbfw = mathbfx","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"Carathéodory's theorem states that we can form a subset of points Q subset P, such that we get a new set of nonnegative weights, mathbfvinmathbbR^N, satisfying","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"mathbfQ mathbfv = beginbmatrix \nvert  vert    vert\nmathbfp_i_1  mathbfp_i_2  cdots  mathbfp_i_N\nvert  vert   vert\nendbmatrix mathbfv = mathbfx = mathbfP mathbfw","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"Once the row indices, i_1 ldots i_N, are sampled, we can obtain the new weights by performing a linear solve on the matrix equation mathbfQ mathbfv = mathbfx.  However, the difficulty in this problem is in subsampling the correct row indices such that the new weights are all nonnegative. The goal of having nonnegative weights can be useful in problems such as numerical quadrature where negative weights could lead to numerical instability. ","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"CaratheodoryPruning.jl implements various algorithms for this row index subselection problem.","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"The base Carathéodory pruning method takes in a matrix V of size M by N, or the transpose of the mathbfP matrix above. It also takes in a vector of nonnegative weights w_in of length M. It then returns a nonnegative pruned vector, w, of length M, and a vector of row indices of length N, inds, such that V[inds,:]' * w[inds] is approximately equal to V' * w_in. If return_errors is set to true, it additionally returns a vector of moment errors at each iteration.","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"caratheodory_pruning","category":"page"},{"location":"index.html#CaratheodoryPruning.caratheodory_pruning","page":"Background","title":"CaratheodoryPruning.caratheodory_pruning","text":"caratheodory_pruning(V, w_in, kernel_downdater, prune_weights![; caratheodory_correction=true, progress=false, zero_tol=1e-16, return_error=false, errnorm=norm])\n\nBase method for Caratheodory pruning of the matrix V and weights w_in. Returns a new set of weights, w, and a set of indices, inds, such that w_in only has nonzero elements at the indices, inds, and\n\nif size(V,1) > size(V,2), Vᵀw_in - V[inds,:]ᵀw_in[inds] ≈ 0\nif size(V,1) < size(V,2), V w_in - V[inds,:] w_in[inds] ≈ 0\n\nUses the kernel_downdater object to generate kernel vectors for pruning, and the prune_weights! method to prune weights after kernel vectors have been formed.\n\nIf caratheodory_correction=true, then uses a linear solve at the end to reduce error in the moments.\n\nIf progress=true, displays a progress bar.\n\nzero_tol determines the tolerance for a weight equaling zero.\n\nIf return_error=true, returns an additional float of moment errors at the end of the procedure.\n\nerrornorm is the method called on the truth moments vs computed moments to evaluate final error, only used if caratheodory_correction=true  or return_error=true. Defaults to LinearAlgebra.jl's norm method.\n\n\n\n\n\ncaratheodory_pruning(V, w_in[; kernel=:GivensUpDown, pruning=:first, caratheodory_correction=true, return_error=false, errnorm=norm, zero_tol=1e-16, progress=false, kernel_kwargs...])\n\nHelper method for calling the base caratheodory_pruning method.\n\nTakes in a symbol for kernel, and forms a KernelDowndater object depending on what is passed in. Also passes additional kwargs into the KernelDowndater:\n\nOptions include :FullQRDowndater or :FullQR, :GivensDowndater or :Givens, :CholeskyDowndater or :Cholesky, :FullQRUpDowndater or :FullQRUpDown, and :GivensUpDownDater or :GivensUpDown.\n\nTakes in a symbol for pruning, and chooses a pruning method depending on what is passed in. Options are :first or :minabs.\n\nSee the other caratheodory_pruning docstring for info on other arguments.\n\n\n\n\n\n","category":"function"},{"location":"index.html","page":"Background","title":"Background","text":"The implemented methods for Carathéodory pruning are iterative kernel-based algorithms. This means that at each step, kernel vectors for the transpose of V[inds,:] are formed so that they can be used to pivot the weights without changing the moments. The pivot is chosen to ensure that (at least) one of the weights are set to zero, and the rest are still nonnegative. This iteration is then repeated until M - N of the row indices are pruned, and we are left with N row indices.","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"Here is a full example of generating a random matrix V and random, positive vector of weights w_in, computing the moments eta, using caratheodory_pruning to generate pruned weights w, and computing the moment error.","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"using CaratheodoryPruning\nusing Random\nusing Random: seed! # hide\nseed!(1) # hide\nM = 100\nN = 10\nV = rand(M, N)\nw_in = rand(M)\neta = V' * w_in\nw, inds = caratheodory_pruning(V, w_in)\nw[inds]","category":"page"},{"location":"index.html","page":"Background","title":"Background","text":"error = maximum(abs.(V[inds,:]' * w[inds] .- eta))","category":"page"}]
}
