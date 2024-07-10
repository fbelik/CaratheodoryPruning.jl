using Revise
using CaratheodoryPruning
using Plots
using ProgressBars
using Random: seed!


opts = (FullQR = Dict(:kernel => :FullQR),
        GivensQR = Dict(:kernel => :Givens),
        Cholesky = Dict(:kernel => :Cholesky),
        FullQRUpDown = Dict(:kernel => :FullQRUpDown),
        GivensUpDown = Dict(:kernel => :GivensUpDown)
)

N = 2 ^ 3
Ms = 2 .^ (6:16)
repeat_each = 4
all_times = [zeros(length(Ms)) for _ in opts]
times = zeros(repeat_each)

# Run each method once first for compilation
M = Ms[1]
V = rand(M, N); w = rand(M);
for (i,opt) in enumerate(opts)
    caratheodory_pruning(V, w; opt...);
end

runmax = 20.0

mean(x) = sum(x) / length(x)
for (i,M) in enumerate(Ms)
    for (j,opt) in enumerate(opts)
        if (all_times[j][i] == 0) && ((i == 1) || (i > 1 && all_times[j][i-1] < runmax))
            for k in ProgressBar(1:repeat_each)
                V = rand(M, N)
                w = rand(M)
                res = @timed caratheodory_pruning(V, w; opt...);
                times[k] = res.time
            end
            all_times[j][i] = mean(times)
            println("Finished method $(keys(opts)[j]) on M=$M/$(Ms[end])")
        elseif (all_times[j][i] == 0)
            all_times[j][i] = all_times[j][i-1]
            println("no longer running method $(keys(opts)[j]) on M=$M/$(Ms[end])")
        end
    end
end
tickvals = -3.0:1.0:6.0
plt = plot(xaxis=:log, yaxis=:log, leg=:topleft, xticks=(10 .^ tickvals, ["10^{$x}" for x in tickvals]), yticks=(10 .^ tickvals, ["10^{$x}" for x in tickvals]))
plot!(plt, Ms,(Ms ./ Ms[3]) .^ 1 .* all_times[5][3], ls=:dash, label="Slope 1")
plot!(plt, Ms,(Ms ./ Ms[3]) .^ 2 .* all_times[3][3], ls=:dash, label="Slope 2")
plot!(plt, Ms,(Ms ./ Ms[3]) .^ 3 .* all_times[2][3], ls=:dash, label="Slope 3")
shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
for (i,method) in enumerate(keys(opts))
    ys = unique(all_times[i])
    xs = Ms[eachindex(ys)]
    scatter!(xs, ys, label=string(method), markershape=shapes[i])
end
times_mat=reduce(hcat, all_times)
ylims!(minimum(times_mat) / 2, maximum(times_mat) + 1e3)
xlabel!("M")
ylabel!("Time (sec)")
title!("Runtime with N=$N, varying M")
savefig(plt, "benchmark.png")