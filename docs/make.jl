using Documenter, ImageReconstruction

makedocs(;
    modules = [ImageReconstruction],
    format = Documenter.HTML(),
    pages = ["Home" => "index.md"],
    repo = "https://github.com/JuliaImages/ImageReconstruction.jl/blob/{commit}{path}#L{line}",
    sitename = "ImageReconstruction.jl",
    assets = String[],
)

deploydocs(; repo = "github.com/JuliaImages/ImageReconstruction.jl")
