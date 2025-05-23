using Documenter
using CaratheodoryPruning

makedocs(
    sitename = "CaratheodoryPruning.jl",
    modules  = [CaratheodoryPruning],
    pages    = [
        "Background" => "index.md",
        "Kernel Downdaters" => "kerneldowndater.md",
        "Pruning" => "pruning.md",
        "On Demand Arrays" => "ondemand.md",
        "MC Example" => "mc_example.md"
    ],
    checkdocs = :none,
    format = Documenter.HTML(prettyurls = false),
    repo = Remotes.GitHub("fbelik", "CaratheodoryPruning.jl")
)

deploydocs(
    repo = "github.com/fbelik/CaratheodoryPruning.jl.git",
)