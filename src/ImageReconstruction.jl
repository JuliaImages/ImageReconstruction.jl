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
    ax1, ax2 = axes(I)

    nax1, nax2 = length(ax1), length(ax2)
    for j in ax2, i in ax1
        x = j - nax2 / 2 + 0.5
        y = i - nax1 / 2 + 0.5
        @inbounds for (k, θₖ) in enumerate(θ)
            t′ = x * cos(θₖ) + y * sin(θₖ)

            a = convert(Int, round((t′ - minimum(t)) / step(t) + 1))

            (a < 1 || a > length(t)) && continue
            α = abs(t′ - t[a])

            I′ = I[i, j]
            P[a, k] += (1 - α) * I′

            (a > length(t) + 1) && continue
            P[a+1, k] += α * I′
        end
    end

    P
end

function _ramp_spatial(N::Int, τ, Npad::Int = N)
    @assert Npad ≥ N
    N2 = N ÷ 2
    hval(n) = n == 0 ? 1 / (4*τ^2) : - mod(n, 2)/(π * n * τ)^2
    [i ≤ N ? hval(i - N2 - 1) : 0. for i = 1:Npad]
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
    I = [zeros(T, pixels, pixels) for _ = 1:nthreads()]
    P′ = [Vector{Complex{T}}(undef, Npad) for _ = 1:nthreads()]
    Q = [Vector{T}(undef, N) for _ = 1:nthreads()]

    l = SpinLock()
    @inbounds @threads for (k, θₖ) in collect(enumerate(θ))
        id = threadid()
        Pid, Qid, Iid = P′[id], Q[id], I[id]

        # filter projection
        Pid[1:N] .= P[:, k]
        Pid[N+1:end] .= 0

        # Need to prevent multiple thread execution during fft/ifft.
        # double free occurs otherwise.
        # https://github.com/JuliaMath/FFTW.jl/issues/134
        lock(l)
        fft!(Pid)
        Pid .*= ramp
        ifft!(Pid)
        unlock(l)
        Qid .= τ .* real.(@view Pid[i:j])

        Qₖ = LinearInterpolation(t, Qid)

        # backproject
        for c in CartesianIndices(first(I))
            x = c.I[2] - pixels ÷ 2 + 0.5
            y = c.I[1] - pixels ÷ 2 + 0.5
            x^2 + y^2 ≥ pixels^2 / 4 && continue
            t′ = x * cos(θₖ) + y * sin(θₖ)
            Iid[c] += Qₖ(t′)
        end
    end
    sum(I) .* π ./ K
end

end # module
