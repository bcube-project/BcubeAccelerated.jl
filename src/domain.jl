
"""
Construct subdomains by cell types using a temporary CPU backend.

This implementation copies the cell type information to the CPU to simplify
the construction of subdomains, then adapts the data back to the provided
backend. This approach is used for performance and compatibility reasons.
"""
function Bcube.build_subdomains_by_celltypes(backend::BcubeBackendAcc, mesh, indices)
    _ctypes = Bcube.cells(mesh)[indices]
    indices_cpu = adapt(KernelAbstractions.get_backend([]), indices)
    _ctypes_cpu = adapt(KernelAbstractions.get_backend([]), _ctypes)
    ctypes_cpu = tuple(unique(_ctypes_cpu)...)
    indice_by_ctypes_cpu = map(
        ct -> indices_cpu[filter(i -> _ctypes_cpu[i] == ct, 1:length(indices_cpu))],
        ctypes_cpu,
    )
    offsets_cpu = Bcube.cumsum_exclusive(map(length, indice_by_ctypes_cpu))
    subdomains_cpu =
        Bcube.SubDomain.(nothing, ctypes_cpu, indice_by_ctypes_cpu, offsets_cpu)
    subdomains = map(x -> adapt(backend, x), subdomains_cpu)
    return subdomains
end

"""
Construct subdomains by face types using a temporary CPU backend.

This implementation copies the cell type information to the CPU to simplify
the construction of subdomains, then adapts the data back to the provided
backend. This approach is used for performance and compatibility reasons.
"""
function Bcube.build_subdomains_by_facetypes(backend::BcubeBackendAcc, mesh, indices)
    indices_cpu = adapt(KernelAbstractions.get_backend([]), indices)
    _ftypes_cpu = adapt(KernelAbstractions.get_backend([]), Bcube.faces(mesh)[indices])
    indices_cpu = adapt(KernelAbstractions.get_backend([]), indices)
    cells_cpu = adapt(KernelAbstractions.get_backend([]), Bcube.cells(mesh))
    f2c_cpu =
        adapt(KernelAbstractions.get_backend([]), Bcube.connectivities_indices(mesh, :f2c))
    f2c_cpu = [f2c_cpu[i] for i in indices_cpu]
    _ftypes_cpu =
        [(_ftypes_cpu[i], cells_cpu[f2c_cpu[i]]...) for i in 1:length(_ftypes_cpu)]
    ftypes_cpu = tuple(unique(_ftypes_cpu)...)
    indice_by_ftypes_cpu = map(
        ft -> indices_cpu[filter(i -> _ftypes_cpu[i] == ft, 1:length(_ftypes_cpu))],
        ftypes_cpu,
    )
    offsets_cpu = Bcube.cumsum_exclusive(map(length, indice_by_ftypes_cpu))
    subdomains_cpu =
        Bcube.SubDomain.(nothing, ftypes_cpu, indice_by_ftypes_cpu, offsets_cpu)
    subdomains = map(x -> adapt(backend, x), subdomains_cpu)
    return subdomains
end

function Bcube._foreach_element(
    f::F,
    domain::D,
    subdomain::SD,
    backend::AbstractBcubeBackendAcc,
) where {F <: Function, D <: Bcube.AbstractDomain, SD <: Bcube.SubDomain}
    indices = Bcube.get_indices(subdomain)
    iter_subdomain = Bcube.SubDomainIterator(domain, subdomain)
    _f(i) = f(iter_subdomain[i]...)
    AK.foreachindex(_f, indices, get_backend(backend))
    KernelAbstractions.synchronize(get_backend(backend))
    return nothing
end

function Bcube._map_element(
    f::F,
    domain::D,
    subdomain::SD,
    backend::AbstractBcubeBackendAcc,
) where {F, D <: Bcube.AbstractDomain, SD <: Bcube.SubDomain}
    error("not implemented")
    # _indices = eachindex(Bcube.get_indices(subdomain))
    # indices = adapt(backend, collect(_indices))
    # iter_subdomain = Bcube.SubDomainIterator(domain, subdomain)
    # # mesh = get_mesh(domain)
    # # test_arg_AK_foreach(get_backend(backend), domain)
    # # test_arg_AK_foreach(get_backend(backend), iter_subdomain)
    # function _f(i)
    #     f(iter_subdomain[i])
    # end
    # a = AK.map(_f, indices, get_backend(backend))
    # KernelAbstractions.synchronize(get_backend(backend))
    # return a
end

function _return_type_size(valSize, f, elementInfo)
    fₑ = Bcube.materialize(f, elementInfo)
    elementPoint = Bcube.get_dummy_element_point(elementInfo)
    value = Bcube.materialize(fₑ, elementPoint)
    _ndims = ndims(value)
    for k in 1:max(1, _ndims)
        valSize[k] = size(value, k)
    end
    value
end

function Bcube.get_return_type_and_codim(
    f::Bcube.AbstractLazy,
    domain::Bcube.AbstractDomain,
    backend::AbstractBcubeBackendAcc,
)
    subdomain = first(Bcube.get_subdomains(domain))
    iter_subdomain = Bcube.SubDomainIterator(domain, subdomain)
    indices = KernelAbstractions.zeros(backend, Int, 1)
    function _f(i, valSize)
        elementInfo = first(iter_subdomain[i])
        value = _return_type_size(valSize, f, elementInfo)
        value
    end
    maxRank = 10
    valSize = KernelAbstractions.zeros(backend, Int, maxRank)
    AK.foreachindex(i -> _f(i, valSize), indices, get_backend(backend))
    KernelAbstractions.synchronize(get_backend(backend))
    ind = findall(x -> x ≠ 0, Array(valSize))
    N = (valSize[ind]...,)
    _T = Base.return_types(i -> _f(i, valSize), (Int,))
    T = eltype(first(_T))
    return T, N
end