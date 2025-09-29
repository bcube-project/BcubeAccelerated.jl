@kernel function KA_assemble_kernel!(y::Y, f::F, V::TV, quadrature, domain, backend::BcubeBackendAcc{KernelKA}) where {Y,F,TV}
    I = @index(Global)
    elementInfo = Bcube._get_index(domain, I)
    vₑ = Bcube.blockmap_shape_functions(V, elementInfo)
    fᵥ = Bcube.materialize(f(vₑ), elementInfo)
    values = Bcube.integrate_on_ref_element(fᵥ, elementInfo, quadrature)
    Bcube._update_b!(y, V, values, elementInfo, domain, backend)
    nothing
end

function AK_assemble_kernel!(i, y::Y, f::F, V::TV, quadrature, domain, backend::BcubeBackendAcc{KernelAK}) where {Y,F,TV}
    elementInfo = Bcube._get_index(domain, i)
    vₑ = Bcube.blockmap_shape_functions(V, elementInfo)
    fᵥ = Bcube.materialize(f(vₑ), elementInfo)
    values = Bcube.integrate_on_ref_element(fᵥ, elementInfo, quadrature)
    Bcube._update_b!(y, V, values, elementInfo, domain, backend)
    nothing
end

function Bcube.__update_b!(b::AbstractVector, dofs, vals, backend::BcubeBackendAcc)
    for (i, val) in zip(dofs, vals)
        Atomix.@atomic b[i] += val
    end
    nothing
end

function Bcube.__update_b!(
    b::AbstractVector,
    idofs,
    intvals::Tuple{Vararg{Tuple,N}},
    backend::BcubeBackendAcc,
) where {N}
    f(x) = Bcube.__update_b!(b, idofs, x, backend)
    map(f, intvals)
    nothing
end

function Bcube.__assemble_linear!(b, f, V, measure::Measure, backend::BcubeBackendAcc{KernelKA})
    KA_assemble!(b, f, nothing, V, measure, backend)
end
function Bcube.__assemble_linear!(b, f, V, measure::Measure, backend::BcubeBackendAcc{KernelAK})
    AK_assemble!(b, f, nothing, V, measure, backend)
end

function KA_assemble!(y, f, elts, V, measure, backend::BcubeBackendAcc{KernelKA})
    quadrature = Bcube.get_quadrature(measure)
    domain = get_domain(measure)
    backendKA = get_backend(backend)
    @assert backendKA == get_backend(y)
    ndrange = length(DomainIterator(domain))
    KA_assemble_kernel!(backendKA, WORKGROUP_SIZE)(
        y,
        f,
        V,
        quadrature,
        domain,
        backend;
        ndrange=ndrange,
    )
    is_sync(backend) && KernelAbstractions.synchronize(backendKA)
    return nothing
end

function AK_assemble!(y, f, elts, V, measure, backend::BcubeBackendAcc{KernelAK})
    quadrature = Bcube.get_quadrature(measure)
    domain = get_domain(measure)
    backendAK = get_backend(backend)
    @assert backendAK == get_backend(y)
    ndrange = length(DomainIterator(domain))
    kernel_f(i) = AK_assemble_kernel!(i, y, f, V, quadrature, domain, backend)
    AK.foreachindex(kernel_f, 1:ndrange, backendAK; block_size=WORKGROUP_SIZE)
    is_sync(backend) && KernelAbstractions.synchronize(backendAK)
    return nothing
end

@kernel function KA_assemble_bilinear_kernel!(
    imat,
    jmat,
    vmat,
    offsets,
    f::F,
    elts::E,
    U::TU,
    V::TV,
    quadrature,
    domain,
    backend::BcubeBackendAcc{KernelKA},
) where {F,E,TU,TV}
    I = @index(Global)
    elementInfo = Bcube._get_index(domain, I)
    λu, λv = Bcube.blockmap_bilinear_shape_functions(U, V, elementInfo)
    g1 = Bcube.materialize(f(λu, λv), elementInfo)
    values = Bcube.integrate_on_ref_element(g1, elementInfo, quadrature)
    Bcube._append_contribution!(
        (offsets[I], vmat),
        imat,
        jmat,
        U,
        V,
        values,
        elementInfo,
        domain,
        backend,
    )
    nothing
end

function AK_assemble_bilinear_kernel!(
    I,
    imat,
    jmat,
    vmat,
    offsets,
    f::F,
    elts::E,
    U::TU,
    V::TV,
    quadrature,
    domain,
    backend::BcubeBackendAcc{KernelAK},
) where {F,E,TU,TV}
    elementInfo = Bcube._get_index(domain, I)
    λu, λv = Bcube.blockmap_bilinear_shape_functions(U, V, elementInfo)
    g1 = Bcube.materialize(f(λu, λv), elementInfo)
    values = Bcube.integrate_on_ref_element(g1, elementInfo, quadrature)
    Bcube._append_contribution!(
        (offsets[I], vmat),
        imat,
        jmat,
        U,
        V,
        values,
        elementInfo,
        domain,
        backend,
    )
    nothing
end

function Bcube._append_bilinear!(I, J, _X::Tuple, row, col, vals, backend::AbstractBcubeBackendAcc)
    offset, X = _X
    _rows, _cols = Bcube._cartesian_product(row, col)
    for k in eachindex(_rows)
        I[offset+k] = _rows[k]
        J[offset+k] = _cols[k]
        X[offset+k] = vals[k]
    end
end

KernelAbstractions.@kernel function _ndofs_element_bilinear_kernel!(
    ndofs,
    U,
    V,
    domain::D,
) where {D<:CellDomain}
    I = @index(Global)
    elementInfo = @inline Bcube._get_index(domain, I)
    nU = Val(Bcube.get_ndofs(U, shape(Bcube.celltype(elementInfo))))
    nV = Val(Bcube.get_ndofs(V, shape(Bcube.celltype(elementInfo))))
    Udofs = Bcube.get_dofs(U, I, nU) # columns correspond to the TrialFunction
    Vdofs = Bcube.get_dofs(V, I, nV) # lines correspond to the TestFunction
    rows, = Bcube._cartesian_product(Vdofs, Udofs)
    ndofs[I] = length(rows)
end

function Bcube.assemble_bilinear!(I, J, X, f, measure::Measure, U, V, backend::BcubeBackendAcc{KernelKA})
    KA_assemble_bilinear(I, J, X, f, nothing, U, V, measure, backend)
end
function Bcube.assemble_bilinear!(I, J, X, f, measure::Measure, U, V, backend::BcubeBackendAcc{KernelAK})
    AK_assemble_bilinear(I, J, X, f, nothing, U, V, measure, backend)
end

function Bcube.allocate_bilinear(backend::AbstractBcubeBackendAcc, a, U, V, T)
    integration = a(Bcube._null_operator(U), Bcube._null_operator(V))
    domain = Bcube.get_domain(Bcube.get_measure(integration))
    backendKA = get_backend(backend)
    ndofs = KernelAbstractions.zeros(backendKA, Int, length(Bcube.indices(domain)))
    _ndofs_element_bilinear_kernel!(backendKA, WORKGROUP_SIZE)(
        ndofs,
        U,
        V,
        domain;
        ndrange=size(ndofs),
    )
    buffersize = AK.reduce(+, ndofs; init=zero(eltype(ndofs)))
    I = KernelAbstractions.zeros(backendKA, Int, buffersize)
    J = KernelAbstractions.zeros(backendKA, Int, buffersize)
    X = KernelAbstractions.zeros(backendKA, Float64, buffersize)
    return I, J, X
end

function KA_assemble_bilinear(I, J, X, f, elts::E, U, V, measure, backend::BcubeBackendAcc{KernelKA}) where {E}
    quadrature = Bcube.get_quadrature(measure)
    domain = get_domain(measure)
    backendKA = get_backend(backend)
    @assert backendKA == get_backend(X)

    ndofs = KernelAbstractions.zeros(backendKA, Int, length(Bcube.indices(domain)))
    _ndofs_element_bilinear_kernel!(backendKA, WORKGROUP_SIZE)(
        ndofs,
        U,
        V,
        domain;
        ndrange=size(ndofs),
    )
    offsets = AK.accumulate(+, ndofs; init=zero(eltype(ndofs)), inclusive=false)
    ndrange = length(DomainIterator(domain))
    KA_assemble_bilinear_kernel!(backendKA, WORKGROUP_SIZE)(
        I,
        J,
        X,
        offsets,
        f,
        elts,
        U,
        V,
        quadrature,
        domain,
        backend;
        ndrange=ndrange,
    )
    is_sync(backend) && KernelAbstractions.synchronize(backendKA)
    return nothing
end


function AK_assemble_bilinear(I, J, X, f, elts::E, U, V, measure, backend::BcubeBackendAcc{KernelAK}) where {E}
    quadrature = Bcube.get_quadrature(measure)
    domain = get_domain(measure)
    backendAK = get_backend(backend)
    @assert backendAK == get_backend(X)

    ndofs = KernelAbstractions.zeros(backendAK, Int, length(Bcube.indices(domain)))
    _ndofs_element_bilinear_kernel!(backendAK, WORKGROUP_SIZE)(
        ndofs,
        U,
        V,
        domain;
        ndrange=size(ndofs),
    )
    offsets = AK.accumulate(+, ndofs; init=zero(eltype(ndofs)), inclusive=false)
    ndrange = length(DomainIterator(domain))
    function kernel_f(i)
        AK_assemble_bilinear_kernel!(i,
            I,
            J,
            X,
            offsets,
            f,
            elts,
            U,
            V,
            quadrature,
            domain,
            backend)
    end
    AK.foreachindex(kernel_f, 1:ndrange, backendAK; block_size=WORKGROUP_SIZE)
    is_sync(backend) && KernelAbstractions.synchronize(backendAK)
    return nothing
end

