module LinearTransportGpu

using Bcube
using BcubeAccelerated
using BcubeVTK
using KernelAbstractions
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using TimerOutputs
using StaticArrays
using Adapt
using BenchmarkTools

const to = TimerOutput()

const VTK_OUTPUT = false

const nx = 500
const ny = 500
const nite = 100
const degree = 1
const c = SA[1.0, 0.0] # Convection velocity (must be a vector)
const CFL = 0.2
const Δt = CFL * min(1.0 / nx, 1.0 / ny) / norm(c) / (2 * degree + 1)
const nout = 20

mutable struct VtkHandler
    basename::Any
    ite::Any
    mesh::Any
    VtkHandler(basename, mesh) = new(basename, 0, mesh)
end

function append_vtk(vtk, u::Bcube.AbstractFEFunction, t)
    # Write
    write_file(
        vtk.basename,
        vtk.mesh,
        Dict("u" => u),
        vtk.ite,
        t;
        discontinuous = true,
        collection_append = vtk.ite > 0,
    )

    # Update counter
    vtk.ite += 1
end

bc_in(t) = PhysicalFunction(x -> c .* cos(3 * x[2])) * sin(4 * reduce(+, t))

function upwind(ui, uj, nij)
    cij = c ⋅ nij
    if cij > zero(cij)
        flux = cij * ui
    else
        flux = cij * uj
    end
    flux
end

_factorize(backend::Bcube.AbstractBcubeBackend, M) = __factorize(get_backend(backend), M)
__factorize(backend, M) = factorize(M)
function __factorize(backend::CUDA.CUDABackend, M)
    F = CUSOLVER.SparseCholesky(M)
    CUSOLVER.spcholesky_factorise(F, M, 1.e-12)
    return F
end

function _solve!(x, A, b, backend::Bcube.AbstractBcubeBackend)
    __solve!(x, A, b, get_backend(backend))
end
function __solve!(x, A, b, backend::CUDA.CUDABackend)
    CUSOLVER.spcholesky_solve(A, b, x)
    return nothing
end
function __solve!(x, A, b, backend)
    x .= A \ b
    return nothing
end

function main(nx, ny, nite, degree, backend)
    mesh_cpu = rectangle_mesh(nx, ny)
    fs = FunctionSpace(:Lagrange, degree)
    U_cpu = TrialFESpace(fs, mesh_cpu, :discontinuous)

    mesh = adapt(backend, mesh_cpu)
    U = adapt(backend, U_cpu)
    V = TestFESpace(U)
    u = FEFunction(U, KernelAbstractions.zeros(backend, Float64, get_ndofs(U)))

    Ω = CellDomain(mesh)
    Γ = InteriorFaceDomain(mesh)
    Γ_in = BoundaryFaceDomain(mesh, (:xmin,))
    Γ_out = BoundaryFaceDomain(mesh, (:xmax, :ymin, :ymax))

    dΩ = Measure(Ω, 2 * degree + 1)
    dΓ = Measure(Γ, 2 * degree + 1)
    dΓ_in = Measure(Γ_in, 2 * degree + 1)
    dΓ_out = Measure(Γ_out, 2 * degree + 1)

    println("Building normals")

    nΓ = get_face_normals(Γ)
    nΓ_in = get_face_normals(Γ_in)
    nΓ_out = get_face_normals(Γ_out)

    println("Building weak forms")

    m(u, v) = ∫(u ⋅ v)dΩ # Mass matrix
    f_Ω(v) = (c * u) ⋅ ∇(v)
    l_Ω(v) = ∫(f_Ω(v))dΩ

    l_Γ(v) = ∫((upwind ∘ (side⁻(u), side⁺(u), side⁻(nΓ))) * jump(v))dΓ
    l_Γ_in(v, t) = ∫((side⁻(bc_in(t)) ⋅ side⁻(nΓ_in)) * side⁻(v))dΓ_in
    l_Γ_in_t2(t) = v -> l_Γ_in(v, t)
    l_Γ_out(v) = ∫((upwind ∘ (side⁻(u), 0.0, side⁻(nΓ_out))) * side⁻(v))dΓ_out

    ## Allocate buffers for linear assembling
    b_vol = Bcube.allocate_dofs(U)
    b_fac = similar(b_vol)
    rhs = similar(b_vol)

    println("Building mass matrix")

    @timeit to "assemble mass matrix" begin
        M = assemble_bilinear(m, U, V)
    end
    @timeit to "factorize mass matrix" begin
        factoM = _factorize(backend, M)
    end

    # factoM = cholesky(M2; check = true)
    #linsolve = LS.init(LS.LinearProblem(CuSparseMatrixCSR(M), b_vol))
    #factoM = cholesky(adapt(backend, Array(M))) # TODO : avoid dense matrix

    println("Starting time loop")

    t = 0.0

    # Write to file
    if VTK_OUTPUT
        out_dir = joinpath(@__DIR__, "myout", "linear_transport")
        mkpath(out_dir)
        vtk = VtkHandler(joinpath(out_dir, "linear_transport.pvd"), mesh_cpu)
        u_cpu = adapt(get_backend(zeros(1)), u)
        append_vtk(vtk, u_cpu, t)
    end

    @timeit to "timeloop" begin
        for i in 1:nite
            (i % nout == 0) && println("$i / $nite")

            ## Reset pre-allocated vectors
            @timeit to "reset buffers" begin
                b_vol .= 0.0
                b_fac .= 0.0
            end

            # CUDA.@profile assemble_linear!(b_vol, l_Ω, V; backend=backend)
            # CUDA.@profile assemble_linear!(b_vol, l_Ω, V; backend=backend)
            # error("ici")

            # Assembling linear form
            @timeit to "assemble linear" begin
                @timeit to "l_Ω" assemble_linear!(b_vol, l_Ω, V)
                @timeit to "l_Γ" assemble_linear!(b_fac, l_Γ, V)
                @timeit to "l_Γ_out" assemble_linear!(b_fac, l_Γ_out, V)
                @timeit to "l_Γ_in" assemble_linear!(b_fac, l_Γ_in_t2(t), V)
            end

            ## Compute rhs
            @timeit to "compute rhs (solve)" begin
                _solve!(rhs, factoM, b_vol - b_fac, backend)
                rhs .*= Δt
            end

            ## Update solution
            @timeit to "update solution" begin
                u.dofValues .+= rhs
            end

            ## Update time
            t += Δt

            ## Write VTK outputs
            if VTK_OUTPUT && ((i % nout) == 0)
                @timeit to "append_vtk" begin
                    u_cpu = adapt(get_backend(zeros(1)), u)
                    append_vtk(vtk, u_cpu, t)
                end
            end
        end
    end
end

#const backendDevice = get_backend(ones(2))  ## CPU
const backendDevice = get_backend(CUDA.ones(2)) ## GPU
const backend = BcubeAccelerated.BcubeBackendAcc(
    backendDevice;
    kernel = BcubeAccelerated.KernelKA(),
    sync = true,
)

#const backend = Bcube.get_bcube_backend()

#warmup :
disable_timer!(to)
main(nx, ny, 1, degree, backend)

#timing :
enable_timer!(to)
main(nx, ny, nite, degree, backend)

show(to)
end