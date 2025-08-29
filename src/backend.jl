abstract type AbstractKernelLib end
struct KernelKA <: AbstractKernelLib end # KernelAbstractions
struct KernelAK <: AbstractKernelLib end # AcceleratedKernels

abstract type AbstractBcubeBackendAcc{K,B,S} <: Bcube.AbstractBcubeBackend end
KernelAbstractions.get_backend(a::AbstractBcubeBackendAcc) = a.backend
is_sync(a::AbstractBcubeBackendAcc) = a.sync


struct BcubeBackendAcc{K,B,S} <: AbstractBcubeBackendAcc{K,B,S}
    backend::B
    sync::S
end

function BcubeBackendAcc(kernel::AbstractKernelLib, backend::KernelAbstractions.Backend, sync::Bool)
    BcubeBackendAcc{typeof(kernel),typeof(backend),typeof(sync)}(backend, sync)
end
BcubeBackendAcc(backend::KernelAbstractions.Backend; kernel=KernelKA(), sync=true) = BcubeBackendAcc(kernel, backend, sync)

KernelAbstractions.allocate(backend::AbstractBcubeBackendAcc, T, dims...) = KernelAbstractions.allocate(get_backend(backend), T, dims...)
KernelAbstractions.synchronize(backend::AbstractBcubeBackendAcc) = KernelAbstractions.synchronize(get_backend(backend))
KernelAbstractions.zeros(backend::AbstractBcubeBackendAcc, T::Type, dims...) = KernelAbstractions.zeros(get_backend(backend), T, dims...)
KernelAbstractions.zeros(backend::AbstractBcubeBackendAcc, ::Type{T}, dims::Tuple) where T = KernelAbstractions.zeros(get_backend(backend), T, dims)
KernelAbstractions.ones(backend::AbstractBcubeBackendAcc, T::Type, dims...) = KernelAbstractions.ones(get_backend(backend), T, dims...)
KernelAbstractions.ones(backend::AbstractBcubeBackendAcc, ::Type{T}, dims::Tuple) where T = KernelAbstractions.ones(get_backend(backend), T, dims)
synchronize(backend::AbstractBcubeBackendAcc) = synchronize(get_backend(backend))



const KABackends = KernelAbstractions.Backend

# type piracy...
KernelAbstractions.allocate(::Bcube.BcubeBackendCPUSerial, T, dims...) = Array{T}(undef, dims)
KernelAbstractions.synchronize(::Bcube.BcubeBackendCPUSerial) = nothing
KernelAbstractions.zeros(backend::Bcube.BcubeBackendCPUSerial, T::Type, dims...) = KernelAbstractions.zeros(backend, T, dims)
KernelAbstractions.zeros(::Bcube.BcubeBackendCPUSerial, ::Type{T}, dims::Tuple) where T = zeros(T, dims)
KernelAbstractions.ones(backend::Bcube.BcubeBackendCPUSerial, T::Type, dims...) = KernelAbstractions.ones(backend, T, dims)
KernelAbstractions.ones(::Bcube.BcubeBackendCPUSerial, ::Type{T}, dims::Tuple) where T = ones(T, dims)
KernelAbstractions.get_backend(a::Bcube.BcubeBackendCPUSerial) = a
is_sync(a::Bcube.BcubeBackendCPUSerial) = false
