module BcubeAccelerated
using Bcube
using Adapt
using GPUArrays
using Atomix
using KernelAbstractions

#using StaticArrays
import AcceleratedKernels as AK

const WORKGROUP_SIZE = 256

include("utils.jl")
include("backend.jl")
include("adapt.jl")
include("fespace.jl")
include("domain.jl")
include("assembler.jl")

end