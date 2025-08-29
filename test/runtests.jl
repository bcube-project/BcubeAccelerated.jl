println("Testing...")
using Bcube
#include(joinpath(@__DIR__, "../src/BcubeAccelerated.jl"))
using BcubeAccelerated
#using BcubeVTK
using KernelAbstractions
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using TimerOutputs

include("linear_transport_gpu.jl")
using .LinearTransportGpu

