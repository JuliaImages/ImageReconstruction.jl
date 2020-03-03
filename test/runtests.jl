using ImageReconstruction
using Test

@testset "radon - centered impulse response" begin
    pixels = 129
    views = 200
    I = zeros(pixels, pixels)
    I[pixels÷2, pixels÷2] = 1
    θ = range(0, step = 2π / views, length = views)
    t = -100:100

    P = radon(I, θ, t)

    @test all(P[101, :] .== 1)
end

using Images: shepp_logan
using Statistics
@testset "iradon - shepp logan" begin
    Igt = shepp_logan(128)
    views = 200
    θ = range(0, 2π, length=views)
    t = -150:150
    P = radon(Igt, θ, t)
    I = iradon(P, θ, t)
    diff = I .- Igt
    @test mean(diff[:]) ≈ 0 atol=1e-3
end
