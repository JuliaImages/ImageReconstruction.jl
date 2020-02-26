using ImageReconstruction
using Test

@testset "radon: pixel-driven" begin
    pixels = 129
    views = 200
    I = zeros(pixels, pixels)
    I[pixels÷2, pixels÷2] = 1
    θ = range(0, step = 2π / views, length = views)
    t = -100:100

    P = radon(I, θ, t)

    @test all(P[101, :] .== 1)
end
