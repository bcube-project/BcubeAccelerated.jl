println("Testing...")
using Adapt
using Bcube
using BcubeAccelerated
using Test
using KernelAbstractions
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using TimerOutputs

@testset "BcubeAccelerated.jl" begin
    if get(ENV, "WITH_CPU_TESTS", "true") == "true"
        include("test_domain.jl")
    end

    if get(ENV, "WITH_CUDA_TESTS", "true") == "true"
        include("linear_transport_gpu.jl")
        using .LinearTransportGpu

        include("covo_gpu.jl")
        using .CovoGpu
    end

    @test true # Checkpoint
end
