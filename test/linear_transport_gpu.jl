module LinearTransportGpu

using Bcube
#include(joinpath(@__DIR__, "../src/BcubeAccelerated.jl"))
using BcubeAccelerated
#using BcubeVTK
using KernelAbstractions
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using TimerOutputs
using StaticArrays
using Adapt
using BcubeVTK
using BenchmarkTools
using Dates

const to = TimerOutput()

const VTK_OUTPUT = false

const nx = 500
const ny = 500
const nite = 100
const degree = 0
const c = SA[1.0, 0.0] # Convection velocity (must be a vector)
const CFL = 0.2
const Δt = CFL * min(1.0 / nx, 1.0 / ny) / norm(c)
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
        discontinuous=true,
        collection_append=vtk.ite > 0,
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

function main(nx, ny, nite, degree, backend)
    mesh_cpu = rectangle_mesh(nx, ny)
    fs = FunctionSpace(:Lagrange, degree)
    U_cpu = TrialFESpace(fs, mesh_cpu, :discontinuous)

    mesh = adapt(backend, mesh_cpu)
    U = adapt(backend, U_cpu)
    V = TestFESpace(U)
    u = FEFunction(U, KernelAbstractions.zeros(backend, Float64, get_ndofs(U)))

    Γ = InteriorFaceDomain(mesh)
    Γ_in = BoundaryFaceDomain(mesh, (:xmin,))
    Γ_out = BoundaryFaceDomain(mesh, (:xmax, :ymin, :ymax))

    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
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
    current_time = KernelAbstractions.zeros(backend, Float64, 1)
    l_Γ_in_t(v) = l_Γ_in(v, current_time)
    l_Γ_in_t2(t) = v -> l_Γ_in(v, t)
    l_Γ_out(v) = ∫((upwind ∘ (side⁻(u), 0.0, side⁻(nΓ_out))) * side⁻(v))dΓ_out

    ## Allocate buffers for linear assembling
    b_vol = KernelAbstractions.ones(backend, Float64, get_ndofs(U))
    b_fac = similar(b_vol)
    rhs = similar(b_vol)

    println("Building mass matrix")

    @timeit to "assemble mass matrix" begin
        M = assemble_bilinear(m, U, V; backend=backend)
    end
    @timeit to "factorize mass matrix" begin
        if isa(backend, CUDA.CUDABackend)
            F = CUSOLVER.SparseCholesky(M)
            CUSOLVER.spcholesky_factorise(F, M, 1.e-12)
        else
            factoM = factorize(M)
        end
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

    t_assemble = zero(Dates.now())

    @timeit to "timeloop" begin
        for i in 1:nite
            (i % nout == 0) && println("$i / $nite")

            ## Reset pre-allocated vectors
            @timeit to "reset buffers" begin
                b_vol .= 0.0
                b_fac .= 0.0
            end

            # Assembling linear form
            @timeit to "assemble linear" begin
                t1 = Dates.now()
                @timeit to "l_Ω" assemble_linear!(b_vol, l_Ω, V; backend=backend)
                @timeit to "l_Γ" assemble_linear!(b_fac, l_Γ, V; backend=backend)
                @timeit to "l_Γ_out" assemble_linear!(b_fac, l_Γ_out, V; backend=backend)
                # tᵢ = copy(t)
                # current_time .= tᵢ
                @timeit to "l_Γ_in" assemble_linear!(b_fac, l_Γ_in_t2(t), V; backend=backend)
                t_assemble += (Dates.now() - t1)
            end
            if false
                @btime assemble_linear!($b_vol, $l_Ω, $V; backend=$backend)
                @btime assemble_linear!($b_fac, $l_Γ, $V; backend=$backend)
                @btime assemble_linear!($b_fac, $l_Γ_out, $V; backend=$backend)
                # tᵢ = copy(t)
                # current_time .= tᵢ
                @btime assemble_linear!($b_fac, $l_Γ_in_t2($t), $V; backend=$backend)
                error("ici")
            end

            ## Compute rhs
            #LS.set_b(linsolve, b_vol - b_fac)
            #sol = LS.solve!(linsolve)
            #rhs .= Δt .* sol.u
            @timeit to "compute rhs (solve)" begin
                if isa(backend, CUDA.CUDABackend)
                    CUSOLVER.spcholesky_solve(F, b_vol - b_fac, rhs)
                else
                    rhs = factoM \ (b_vol - b_fac)
                end
                rhs .*= Δt
            end

            ## Update solution
            @timeit to "update solution" begin
                u.dofValues .+= rhs
            end

            ## Update time
            t += Δt

            if VTK_OUTPUT && ((i % nout) == 0)
                @timeit to "append_vtk" begin
                    u_cpu = adapt(get_backend(zeros(1)), u)
                    append_vtk(vtk, u_cpu, t)
                end
            end
        end
    end
    @show t_assemble
end

#const backend = get_backend(ones(2))  ## CPU
const backend = get_backend(CUDA.ones(2)) ## GPU
#const backend = BcubeAccelerated.DefaultBcubeCPUBackend()


#warmup :
disable_timer!(to)
main(nx, ny, 1, degree, backend)

#timing :
enable_timer!(to)
main(nx, ny, nite, degree, backend)

show(to)
end