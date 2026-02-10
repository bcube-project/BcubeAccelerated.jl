println("Testing...")
using Bcube
using BcubeAccelerated
#using BcubeVTK
using KernelAbstractions
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using TimerOutputs

@testset "BcubeAccelerated.jl" begin

    include("linear_transport_gpu.jl")
    using .LinearTransportGpu

    include("covo_gpu.jl")
    using .CovoGpu

end
