using Documenter
using CaratheodoryPruning

makedocs(
    sitename = "CaratheodoryPruning.jl",
    modules  = [CaratheodoryPruning],
    pages    = [
        "Background" => "index.md",
        "Kernel Downdaters" => "kerneldowndater.md",
        "Pruning" => "pruning.md",
    ],
    checkdocs = :none,
    format = Documenter.HTML(prettyurls = false),
    #repo = Remotes.GitHub("fbelik", "CaratheodoryPruning.jl")
)