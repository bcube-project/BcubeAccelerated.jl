println("Testing...")
using Adapt
using Bcube
using BcubeAccelerated
using Test
using KernelAbstractions
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using TimerOutputs

WITH_CPU_TESTS = get(ENV, "WITH_CPU_TESTS", "true") == "true"
WITH_GPU_TESTS = get(ENV, "WITH_CUDA_TESTS", "true") == "true"

@show WITH_CPU_TESTS, WITH_CUDA_TESTS

@testset "BcubeAccelerated.jl" begin
    if WITH_CPU_TESTS
        include("test_domain.jl")
    end

    if WITH_CUDA_TESTS
        include("linear_transport_gpu.jl")
        using .LinearTransportGpu

        include("covo_gpu.jl")
        using .CovoGpu
    end

    @test true # Checkpoint
end
