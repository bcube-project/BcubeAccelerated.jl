function allocate_dofs(
    backend::AbstractBcubeBackendAcc,
    feSpace::Union{SingleFESpace, MultiFESpace},
    T = Float64,
)
    KernelAbstractions.zeros(backend, T, get_ndofs(feSpace))
end
