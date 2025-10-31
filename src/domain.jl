
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
    indice_by_ctypes_cpu =
        map(ct -> indices_cpu[filter(i -> _ctypes_cpu[i] == ct, 1:length(indices_cpu))], ctypes_cpu)
    subdomains_cpu = Bcube.SubDomain.(nothing, ctypes_cpu, indice_by_ctypes_cpu)
    subdomains = map(x -> adapt(backend, x), subdomains_cpu)
    return subdomains
end

function Bcube.build_subdomains_by_facetypes(backend::BcubeBackendAcc, mesh, indices)
    indices_cpu = adapt(KernelAbstractions.get_backend([]), indices)
    _ftypes_cpu = adapt(KernelAbstractions.get_backend([]), Bcube.faces(mesh)[indices])
    indices_cpu = adapt(KernelAbstractions.get_backend([]), indices)
    cells_cpu = adapt(KernelAbstractions.get_backend([]), Bcube.cells(mesh))
    f2c_cpu = adapt(KernelAbstractions.get_backend([]), Bcube.connectivities_indices(mesh, :f2c))
    f2c_cpu = [f2c_cpu[i] for i in indices_cpu]
    _ftypes_cpu = [(_ftypes_cpu[i], cells_cpu[f2c_cpu[i]]...) for i in 1:length(_ftypes_cpu)]
    ftypes_cpu = tuple(unique(_ftypes_cpu)...)
    indice_by_ftypes_cpu =
        map(ft -> indices_cpu[filter(i -> _ftypes_cpu[i] == ft, 1:length(_ftypes_cpu))], ftypes_cpu)
    subdomains_cpu = Bcube.SubDomain.(nothing, ftypes_cpu, indice_by_ftypes_cpu)
    subdomains = map(x -> adapt(backend, x), subdomains_cpu)
    return subdomains
end



function Bcube._foreach_element(f, domain::AbstractDomain, subdomain, backend::AbstractBcubeBackendAcc)
    indices = Bcube.get_indices(subdomain)
    iter_subdomain = Bcube.SubDomainIterator(domain, subdomain)
    _f(i) = f(iter_subdomain[i])
    AK.foreachindex(_f, indices, get_backend(backend))
    KernelAbstractions.synchronize(get_backend(backend))
    return nothing
end


function Bcube._map_element(f, domain::AbstractDomain, subdomain, backend::AbstractBcubeBackendAcc)
    indices = eachindex(Bcube.get_indices(subdomain))
    iter_subdomain = Bcube.SubDomainIterator(domain, subdomain)
    _f(i) = f(iter_subdomain[i])
    a = AK.map(_f, indices, get_backend(backend))
    KernelAbstractions.synchronize(get_backend(backend))
    return a
end