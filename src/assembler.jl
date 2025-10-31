
function Bcube.__update_b!(b::AbstractVector, dofs, vals, backend::BcubeBackendAcc)
    for (i, val) in zip(dofs, vals)
        Atomix.@atomic b[i] += val
    end
    nothing
end

function Bcube.__update_b!(
    b::AbstractVector,
    idofs,
    intvals::Tuple{Vararg{Tuple, N}},
    backend::BcubeBackendAcc,
) where {N}
    f(x) = Bcube.__update_b!(b, idofs, x, backend)
    map(f, intvals)
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
            X[offset + k] = vij
        end
    end
end

function _ndofs_element_bilinear_kernel!(ndofs, elementInfo::CellInfo, U, V)
    I = get_element_index(elementInfo)
    nU = Val(Bcube.get_ndofs(U, shape(Bcube.celltype(elementInfo))))
    nV = Val(Bcube.get_ndofs(V, shape(Bcube.celltype(elementInfo))))
    Udofs = Bcube.get_dofs(U, I, nU) # columns correspond to the TrialFunction
    Vdofs = Bcube.get_dofs(V, I, nV) # lines correspond to the TestFunction
    rows, = Bcube._cartesian_product(Vdofs, Udofs)
    ndofs[I] = length(rows)
end

function Bcube.allocate_bilinear(backend::AbstractBcubeBackendAcc, a, U, V, T)
    integration = a(Bcube._null_operator(U), Bcube._null_operator(V))
    domain = Bcube.get_domain(Bcube.get_measure(integration))
    backendKA = get_backend(backend)
    ndofs = KernelAbstractions.zeros(get_backend(backend), Int, Bcube.get_nelements(domain))
    Bcube.foreach_element(e -> _ndofs_element_bilinear_kernel!(ndofs, e, U, V), domain)
    buffersize = AK.reduce(+, ndofs; init = zero(eltype(ndofs)))
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
