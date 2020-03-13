module ImageReconstruction

using Base.Threads
using Interpolations
using FFTW

export radon, iradon

"""
    radon(I::AbstractMatrix, θ::AbstractRange, t::AbstractRange)

Radon transform of a image `I` producing a sinogram from view angles `θ` in
radians and detector sampling `t`.
"""
function radon(I::AbstractMatrix, θ::AbstractRange, t::AbstractRange)
    P = zeros(eltype(I), length(t), length(θ))
    nr, nc = size(I)

    for i = 1:nr, j = 1:nc
        x = j - nc / 2 + 0.5
        y = i - nr / 2 + 0.5
        @inbounds for (k, θₖ) in enumerate(θ)
            t′ = x * cos(θₖ) + y * sin(θₖ)

            a = convert(Int, round((t′ - minimum(t)) / step(t) + 1))

            (a < 1 || a > length(t)) && continue
            α = abs(t′ - t[a])
            P[a, k] += (1 - α) * I[i, j]

            (a > length(t) + 1) && continue
            P[a+1, k] += α * I[i, j]
        end
    end

    P
end

function _ramp_spatial(N::Int, τ, Npad::Int = N)
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
	iradon(P::AbstractMatrix, θ::AbstractRange, t::AbstractRange)

Inverse radon transform of a sinogram `P` with view angles `θ` in radians and
detector sampling `t` producing an image on a 128x128 matrix.
"""
function iradon(P::AbstractMatrix, θ::AbstractRange, t::AbstractRange)
    pixels = 128

    N = length(t)
    K = length(θ)
    Npad = nextpow(2, 2 * N - 1)
    τ = step(t)
    ramp = fft(_ramp_spatial(N, τ, Npad))
    i = N ÷ 2 + 1
    j = i + N - 1

    T = eltype(P)
    image = [zeros(T, pixels, pixels) for _ = 1:nthreads()]
    P′ = [Vector{Complex{T}}(undef, Npad) for _ = 1:nthreads()]
    Q = [Vector{T}(undef, N) for _ = 1:nthreads()]

    l = SpinLock()
    @inbounds @threads for (k, θₖ) in collect(enumerate(θ))
        id = threadid()

        # filter projection
        P′[id][1:N] .= P[:, k]
        P′[id][N+1:end] .= 0

        # Need to prevent multiple thread execution during fft/ifft.
        # double free occurs otherwise.
        # https://github.com/JuliaMath/FFTW.jl/issues/134
        lock(l)
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
            x^2 + y^2 ≥ pixels^2 / 4 && continue
            t′ = x * cos(θₖ) + y * sin(θₖ)
            image[id][c] += Qₖ(t′)
        end
    end
    sum(image) .* π ./ K
end

end # module
