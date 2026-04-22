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
    include("test_domain.jl")

    include("linear_transport_gpu.jl")
    using .LinearTransportGpu

    include("covo_gpu.jl")
    using .CovoGpu

    @test true # Checkpoint
end
