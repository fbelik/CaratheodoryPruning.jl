using CaratheodoryPruning
using Plots
using ColorSchemes
colors = palette(:tab10)
using Random: seed!
seed!(1)

Ns = 2:7
sizes = [(48,10),
         (246,21),
         (642,36),
         (2352,55),
         (4800,78),
         (8427,105)
]

opts = (FullQR = Dict(:kernel => :FullQR),
        GivensQR = Dict(:kernel => :Givens),
        Cholesky_00pctQR = Dict(:kernel => :Cholesky, :pct_full_qr => 0.0),
        Cholesky_05pctQR = Dict(:kernel => :Cholesky, :pct_full_qr => 5.0),
        Cholesky_10pctQR = Dict(:kernel => :Cholesky, :pct_full_qr => 10.0),
        FullQRUpDown = Dict(:kernel => :FullQRUpDown),
        GivensUpDown_00pctQR = Dict(:kernel => :GivensUpDown, :pct_full_qr => 0.0),
        GivensUpDown_01pctQR = Dict(:kernel => :GivensUpDown, :pct_full_qr => 1.0),
        GivensUpDown_02pctQR = Dict(:kernel => :GivensUpDown, :pct_full_qr => 2.0)
)

Ns = 2:5 # Uncomment for shorter runtime
for (N,size) in zip(Ns, sizes)
    V =  rand(size...)
    w_in = rand(size[1])
    ymin = 1e-16
    ymax = 1e-16
    plt = plot(ylims=(1e-16, 1e-10), yticks=10.0 .^ (-16:2:2), yaxis=:log, leg=:outerright, title="Accumulated errors N=$N", size=(800,400))
    for (im,method) in enumerate(keys(opts))
        _, _, errs = caratheodory_pruning(V, w_in, return_errors=true, caratheodory_correction=true, progress=true; opts[method]...)
        if (minimum(errs) / 10) > ymin
            ymin = minimum(errs) / 10
        end
        if maximum(errs) > ymax
            ymax = maximum(errs)
        end
        plot!(plt, max.(errs, 1e-16), label="$method", c=colors[im])
        scatter!(plt, [length(errs)], max.(errs[end:end], 1e-16), c=colors[im], label=false)

        println("Finished method $method on N=$N")
    end
    ylims!(ymin, ymax)
    yticks!(10.0 .^ (floor(Int, log10(ymin)):ceil(Int, log10(ymax))))
    ylabel!("Accumulated Moment Error")
    xlabel!("Iteration")
    savefig(plt, "errs_N$N.png")
end