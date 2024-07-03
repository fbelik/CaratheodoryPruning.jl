using CaratheodoryPruning
using Plots

Ns = 2:7
sizes = [(48,10),
         (246,21),
         (642,36),
         (2352,55),
         (4800,78),
         (8427,105)
]

opts = (FullQR = Dict(:kernel => :FullQR),
        GivensQR = Dict(:kernel => :GivensQR),
        Cholesky_00pctQR = Dict(:kernel => :Cholesky, :pct_full_qr => 0.0),
        Cholesky_05pctQR = Dict(:kernel => :Cholesky, :pct_full_qr => 5.0),
        Cholesky_10pctQR = Dict(:kernel => :Cholesky, :pct_full_qr => 10.0),
        CholeskyUpDown_00pctQR = Dict(:kernel => :CholeskyUpDown, :pct_full_qr => 0.0),
        CholeskyUpDown_05pctQR = Dict(:kernel => :CholeskyUpDown, :pct_full_qr => 5.0),
        CholeskyUpDown_10pctQR = Dict(:kernel => :CholeskyUpDown, :pct_full_qr => 10.0),
)

# Ns = 2:5 # Uncomment for shorter runtime
for (N,size) in zip(Ns, sizes)
    V =  rand(size...)
    w_in = rand(size[1])
    plt = plot(ylims=(1e-16, 1e-8), yaxis=:log, leg=:topleft)
    for (im,method) in enumerate(keys(opts))
        _, _, errs = caratheodory_pruning(V, w_in, return_errors=true; opts[method]...)
        plot!(plt, max.(errs, 1e-16), label="$method")
        println("Finished method $method on N=$N")
    end
    savefig(plt, "errs_N$N.png")
end