# abstract type AbstractBcubeBackendKA{T} <: Bcube.AbstractBcubeBackend{T} end

# struct BcubeBackendKA{T} <: AbstractBcubeBackendKA{T}
#     backend::T
# end

# BcubeBackendCUDA() = BcubeBackendKA(CUDA.CUDABackend())
# BcubeBackendCPUThreaded() = BcubeBackendKA(KernelAbstractions.CPU())

# Adapt.adapt(backend::AbstractBcubeBackendKA, a) = adapt(backend.backend, a)



# abstract type AbstractBcubeBackendStyleKA <: Bcube.AbstractBcubeBackendStyle end
# struct CUDABackendStyle <: AbstractBcubeBackendStyleKA end
# struct CPUThreadBackendStyle <: AbstractBcubeBackendStyleKA end
# backend_style(::CUDA.CUDABackend) = CUDABackendStyle()
# backend_style(::KernelAbstractions.CPU) = CPUThreadBackendStyle()


const KABackends = KernelAbstractions.Backend

struct DefaultBcubeCPUBackend end

KernelAbstractions.allocate(::DefaultBcubeCPUBackend, T, dims...) = Array{T}(undef, dims)
KernelAbstractions.synchronize(::DefaultBcubeCPUBackend) = nothing
KernelAbstractions.zeros(backend::DefaultBcubeCPUBackend, T::Type, dims...) = KernelAbstractions.zeros(backend, T, dims)
KernelAbstractions.zeros(::DefaultBcubeCPUBackend, ::Type{T}, dims::Tuple) where T = zeros(T, dims)
KernelAbstractions.ones(backend::DefaultBcubeCPUBackend, T::Type, dims...) = KernelAbstractions.ones(backend, T, dims)
KernelAbstractions.ones(::DefaultBcubeCPUBackend, ::Type{T}, dims::Tuple) where T = ones(T, dims)
synchronize(::DefaultBcubeCPUBackend) = nothing