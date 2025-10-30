
"""
    build_subdomains_by_celltypes(mesh, indices)

Construct sub‑domains of a mesh grouped by cell type.

# Arguments
- `mesh`: the mesh from which cells are taken.
- `indices`: a collection of cell indices to be considered.

# Returns
A tuple of `SubDomain` objects, each containing:
- a tag to identify groups of `SubDomain` (`tag=nothing` by default),
- the cell type,
- the list of indices belonging to that type.
"""
# function build_subdomains_by_celltypes(mesh, indices::CuArray)
#     _ctypes = cells(mesh)[indices]
#     ctypes = tuple(unique(_ctypes)...)
#     indice_by_ctypes =
#         map(ct -> indices[filter(i -> _ctypes[i] == ct, 1:length(indices))], ctypes)
#     return SubDomain.(nothing, ctypes, indice_by_ctypes)
# end


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