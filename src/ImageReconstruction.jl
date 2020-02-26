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

    for i = 1:nr, j = 1:nc, (k, θₖ) in enumerate(θ)
        x = j - nc / 2 + 0.5
        y = i - nr / 2 + 0.5
        t′ = x * cos(θₖ) + y * sin(θₖ)

        #TODO this could probably be changed to O(1)
        #TODO need to bounds check a and a+1
        a = findfirst(x -> abs(x - t′) < step(t), t)

        α = abs(t′ - t[a])
        P[a, k] += (1 - α) * image[i, j]
        P[a+1, k] += α * image[i, j]
    end

    P
end

end # module
