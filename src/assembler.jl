
function Bcube.__update_b!(
    b::AbstractVector{T},
    dofs::AbstractVector{<:Integer},
    vals::NTuple{N, T},
    backend::BcubeBackendAcc,
) where {N, T}
    for (i, val) in zip(dofs, vals)
        Atomix.@atomic b[i] += val
    end
    nothing
end

function Bcube._append_bilinear!(
    I,
    J,
    X,
    offset,
    row,
    col,
    vals,
    backend::AbstractBcubeBackendAcc,
)
    _rows, _cols = Bcube._cartesian_product(row, col)
    for k in eachindex(_rows)
        I[offset + k] = _rows[k]
        J[offset + k] = _cols[k]
    end
    k = 0
    for vi in vals
        for vij in vi
            k += 1
            Atomix.@atomic X[offset + k] += vij
        end
    end
end

function _ndofs_element_bilinear_kernel!(ndofs, elementInfo::CellInfo, U, V)
    I = get_element_index(elementInfo)
    nU = Bcube.get_ndofs(U, shape(Bcube.celltype(elementInfo)))
    nV = Bcube.get_ndofs(V, shape(Bcube.celltype(elementInfo)))
    ndofs[I] = nU * nV
end

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
        Bcube.foreach_element(e -> _ndofs_element_bilinear_kernel!(ndofs, e, U, V), dom)
        AK.reduce(+, ndofs; init = zero(eltype(ndofs)))
    end

    I = KernelAbstractions.zeros(backendKA, Int, buffersize)
    J = KernelAbstractions.zeros(backendKA, Int, buffersize)
    X = KernelAbstractions.zeros(backendKA, T, buffersize)
    return I, J, X
end

function Bcube._offsets_bilinear_contribution(U, V, domain, backend::BcubeBackendAcc)
    ndofs = KernelAbstractions.zeros(get_backend(backend), Int, Bcube.get_nelements(domain))
    Bcube.foreach_element(e -> _ndofs_element_bilinear_kernel!(ndofs, e, U, V), domain)
    offsets = AK.accumulate(+, ndofs; init = zero(eltype(ndofs)), inclusive = false)
    return offsets
end

function Bcube.allocate_linear(backend::AbstractBcubeBackendAcc, V, T)
    KernelAbstractions.zeros(get_backend(backend), T, Bcube.get_ndofs(V))
end
