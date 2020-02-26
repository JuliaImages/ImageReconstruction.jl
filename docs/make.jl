using Documenter, ImageReconstruction

makedocs(;
    modules=[ImageReconstruction],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/kczimm/ImageReconstruction.jl/blob/{commit}{path}#L{line}",
    sitename="ImageReconstruction.jl",
    authors="Kevin C. Zimmerman",
    assets=String[],
)

deploydocs(;
    repo="github.com/kczimm/ImageReconstruction.jl",
)
