module ImageReconstruction

using FFTW

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

function _ramp_spatial(N::Int, τ)
    h = zeros(N)
    N2 = N ÷ 2
    for i in eachindex(h)
        n = i - N2 - 1
        if mod(n, 2) != 0
            h[i] = -1 / (π * n * τ)^2
        elseif n == 0
            h[i] = 1 / (4 * τ^2)
        end
    end
    h
end

_zero_pad(p::AbstractVector, N::Int) = vcat(p, zeros(N))

function iradon(sinogram::AbstractMatrix, θ::AbstractRange, t::AbstractRange)
    pixels = 128

    N = length(t)
    K = length(θ)
    Npad = nextpow(2, 2 * N - 1)
    τ = step(t)
    h = _ramp_spatial(N, τ)

    # filter sinogram
    Q = similar(sinogram)
    j = div(N, 2) + 1
    k = j + N - 1
    for i in eachindex(θ)
        Q[:, i] .=
            τ .*
            real.(ifft(
                fft(_zero_pad(sinogram[:, i], Npad - N)) .*
                fft(_zero_pad(h, Npad - N)),
            )[j:k])
    end

    image = zeros(eltype(sinogram), pixels, pixels)
    for c in CartesianIndices(image)
        x = c.I[1] - pixels ÷ 2 + 0.5
        y = c.I[2] - pixels ÷ 2 + 0.5
        image[i] = sum(θ) do θᵢ
            t′ = x * cos(θᵢ) + y * sin(θᵢ)
        end
    end
    @. image * π / K
end

end # module
