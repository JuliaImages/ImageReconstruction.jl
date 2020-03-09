module ImageReconstruction

using Base.Threads
using Interpolations
using FFTW

export radon, iradon

"""
radon transform

https://en.wikipedia.org/wiki/Radon_transform
"""
function radon end

"""
inverse radon transform

https://en.wikipedia.org/wiki/Radon_transform
"""
function iradon end


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

            a = convert(Int, round((t′ - minimum(t)) / step(t) + 1))

            (a < 1 || a > length(t)) && continue
            α = abs(t′ - t[a])
            P[a, k] += (1 - α) * image[i, j]

            (a > length(t) + 1) && continue
            P[a+1, k] += α * image[i, j]
        end
    end

    P
end

function _ramp_spatial(N::Int, τ, Npad::Int=N)
    h = zeros(Npad)
    N2 = N ÷ 2
    for i = 1:N
        n = i - N2 - 1
        if mod(n, 2) != 0
            h[i] = -1 / (π * n * τ)^2
        elseif n == 0
            h[i] = 1 / (4 * τ^2)
        end
    end
    h
end


"""
Inverse radon transform using a ramp filter and a pixel-driven algorithm.
"""
function iradon(sinogram::AbstractMatrix, θ::AbstractRange, t::AbstractRange)
    pixels = 128

    N = length(t)
    K = length(θ)
    Npad = nextpow(2, 2 * N - 1)
    τ = step(t)
    ramp = fft(_ramp_spatial(N, τ, Npad))
    i = div(N, 2) + 1
    j = i + N - 1

	T = eltype(sinogram)
	image = [zeros(T, pixels, pixels) for _ = 1:nthreads()]
	P′ = [Vector{Complex{T}}(undef, Npad) for _ in 1:nthreads()]
	Q = [Vector{T}(undef, N) for _ in 1:nthreads()]

	l = SpinLock()
	@inbounds @threads for (k, θₖ) in collect(enumerate(θ))
		id = threadid()

        # filter projection
		P′[id][1:N] .= sinogram[:, k]
		P′[id][N+1:end] .= 0

		lock(l) # threading issues with FFTW (#134)?
		fft!(P′[id])
		P′[id] .*= ramp
		ifft!(P′[id])
		unlock(l)
		Q[id] .= τ .* real.(@view P′[id][i:j])

		Qₖ = LinearInterpolation(t, Q[id])

        # backproject
		for c in CartesianIndices(first(image))
            x = c.I[2] - pixels ÷ 2 + 0.5
            y = c.I[1] - pixels ÷ 2 + 0.5
            x^2+y^2 ≥ pixels^2/4 && continue
            t′ = x * cos(θₖ) + y * sin(θₖ)
			image[id][c] += Qₖ(t′)
        end
    end
	sum(image) .* π ./ K
end

end # module
