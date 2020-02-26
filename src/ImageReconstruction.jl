module ImageReconstruction

export radon, iradon

"""
radon transform

https://en.wikipedia.org/wiki/Radon_transform
"""
radon

"""
inverse radon transform

https://en.wikipedia.org/wiki/Radon_transform
"""
iradon


"""
Radon transform of an image using a pixel-driven algorithm.
"""
function radon(image::AbstractMatrix, θ::AbstractRange, t::AbstractRange)
    P = zeros(eltype(image), length(t), length(θ))
    nr, nc = size(image)

    for i = 1:nr, j = 1:nc
        x = j - nc / 2 + 0.5
        y = i - nr / 2 + 0.5
        @inbounds for (k, θₖ) in enumerate(θ)
            t′ = x * cos(θₖ) + y * sin(θₖ)

            a = convert(Int, round((t′ - t.start) / step(t) + 1))

            (a < 1 || a > length(t)) && continue
            α = abs(t′ - t[a])
            P[a, k] += (1 - α) * image[i, j]

            (a > length(t) + 1) && continue
            P[a+1, k] += α * image[i, j]
        end
    end

    P
end

end # module
