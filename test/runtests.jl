println("Testing...")
using Adapt
using Bcube
using BcubeAccelerated
using Test
using KernelAbstractions
using SparseArrays, LinearAlgebra

WITH_CPU_TESTS = get(ENV, "WITH_CPU_TESTS", "true") == "true"
WITH_CUDA_TESTS = get(ENV, "WITH_CUDA_TESTS", "true") == "true"

if WITH_CUDA_TESTS
    using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
    using TimerOutputs
end

@show WITH_CPU_TESTS, WITH_CUDA_TESTS

@testset "BcubeAccelerated.jl" begin
    if WITH_CPU_TESTS
        include("test_domain.jl")
        include("test_dirichlet.jl")
        include("test_stokes.jl")
    end

    if WITH_CUDA_TESTS
        include("linear_transport_gpu.jl")
        using .LinearTransportGpu

        include("covo_gpu.jl")
        using .CovoGpu

        include("test_dirichlet_gpu.jl")
        include("test_stokes_gpu.jl")
    end

    @test true # Checkpoint
end
