
function Bcube.__update_b!(
    b::AbstractVector{T1},
    dofs::AbstractVector{<:Integer},
    vals::NTuple{N, T2},
    backend::BcubeBackendAcc,
) where {T1 <: Number, T2 <: Number, N}
    for (i, val) in zip(dofs, vals)
        Atomix.@atomic b[i] += val
    end
    nothing
end

# #fix ambiguity
# function Bcube.__update_b!(
#     b::AbstractVector,
#     dofs,
#     vals::NullOperator,
#     backend::BcubeBackendAcc,
# )
#     nothing
# end

function Bcube.allocate_bilinear(
    backend::AbstractBcubeBackendAcc,
    domain::Tuple{Vararg{AbstractDomain}},
    U,
    V,
    T,
)
    backendKA = get_backend(backend)
    buffersize = sum(domain) do dom
        ndofs = KernelAbstractions.zeros(get_backend(backend), Int, Bcube.get_nelements(dom))
        Bcube.foreach_element(
            (e, i, _) -> Bcube._nnz_bilinear_by_element!(ndofs, i, e, U, V),
            dom,
        )
        AK.reduce(+, ndofs; init = zero(eltype(ndofs)))
    end

    I = KernelAbstractions.ones(backendKA, Int, buffersize)
    J = KernelAbstractions.ones(backendKA, Int, buffersize)
    X = KernelAbstractions.zeros(backendKA, T, buffersize)
    return I, J, X
end

function Bcube._offsets_bilinear_contribution(U, V, domain, backend::BcubeBackendAcc)
    ndofs = KernelAbstractions.zeros(get_backend(backend), Int, Bcube.get_nelements(domain))
    Bcube.foreach_element(
        (e, i, _) -> Bcube._nnz_bilinear_by_element!(ndofs, i, e, U, V),
        domain,
    )
    offsets = AK.accumulate(+, ndofs; init = zero(eltype(ndofs)), inclusive = false)
    ndofs_tot = AK.reduce(+, ndofs; init = zero(eltype(ndofs)))
    return offsets, ndofs_tot
end

function Bcube.allocate_linear(backend::AbstractBcubeBackendAcc, V, T)
    KernelAbstractions.zeros(get_backend(backend), T, Bcube.get_ndofs(V))
end
