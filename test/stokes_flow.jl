module stokes_flow_API #hide
println("Running Stokes flow API example...") #hide

# import necessary packages
using BcubeAccelerated
using Bcube
using BcubeGmsh
using BcubeVTK
using LinearAlgebra
using StaticArrays
using KernelAbstractions, Adapt

@show Base.Threads.nthreads()

# Function space: P2 for velocity and P1 for pressure
const fspace = :Lagrange
const degree_u = 2
const degquad = 2 * degree_u + 1
const degree_p = 1

# Material parameters
const ρ = 1400.0
const μ = 14.0
const ν = μ / ρ

# Parameters for the oscillating plate case
const f = 1.0
const k = sqrt(π * f / ν)

const Δt = 1.0e-3
const finalTime = 2.0

const outputpath = @__DIR__
mkpath(outputpath)

# Function that runs the steady case:
"""
    run_steady(; accelerated = true, export_vtk = true, backend_device = CPU(), solve = (\\))

Solve the steady Stokes system on sequential or accelerated CPU, optionally exporting VTK.

Return the lifted solution, constrained system and prescribed values for validation.
Pass `backend_device` and `solve(matrix, rhs)` to use a different execution backend
and solve the constrained system with its sparse solver.
"""
function run_steady(; accelerated = true, export_vtk = true, backend_device = CPU(), solve = (\))
    backend = BcubeAccelerated.BcubeBackendAcc(
        backend_device;
        kernel = BcubeAccelerated.KernelKA(),
        sync = true,
    )

    # Read 2D mesh
    mesh_path = joinpath(@__DIR__, "assets", "domainTriangle_tri.msh")
    mesh_cpu = read_mesh(mesh_path)

    # Definition of trial and test function spaces (with associated Dirichlet boundary conditions)
    fsu = FunctionSpace(fspace, degree_u)
    stationary_wall = PhysicalFunction(_ -> SA[0.0, 0.0])
    moving_lid = PhysicalFunction(_ -> SA[1.0, 0.0])

    # Apply the lid first so the stationary walls prescribe the shared corner dofs.
    velocity_conditions = (
        edge_top = _ -> moving_lid,
        edge_left = _ -> stationary_wall,
        edge_right = _ -> stationary_wall,
    )
    velocity_space =
        Bcube.SingleFESpace(fsu, mesh_cpu, keys(velocity_conditions); size = 2)
    U_vel_cpu = TrialFESpace(velocity_space, velocity_conditions)
    V_vel_cpu = TestFESpace(U_vel_cpu)

    fsp = FunctionSpace(fspace, degree_p)
    U_pre_cpu = TrialFESpace(fsp, mesh_cpu, Dict("edge_top" => SA[0.0]); size = 1)
    V_pre_cpu = TestFESpace(U_pre_cpu)

    # Define MultiFESpace (mixed formalism)
    U_cpu = MultiFESpace(U_vel_cpu, U_pre_cpu)
    V_cpu = MultiFESpace(V_vel_cpu, V_pre_cpu)

    # Adapt
    mesh = accelerated ? adapt(backend, mesh_cpu) : mesh_cpu
    U = accelerated ? adapt(backend, U_cpu) : U_cpu
    V = accelerated ? adapt(backend, V_cpu) : V_cpu

    # Define measures for cell
    dΩ = Measure(CellDomain(mesh), degquad)

    # volume force is equal to zero
    f = PhysicalFunction(x -> SA[0.0, 0.0])

    # definition of bilinear and linear forms
    a((u, p), (v, q)) = ∫(μ * ∇(u) ⊡ ∇(v) - tr(∇(v)) * p + tr(∇(u)) * q)dΩ
    l((v, q)) = ∫(f ⋅ v)dΩ

    # assemble matrix and RHS
    A = assemble_bilinear(a, U, V)
    L = assemble_linear(l, V)

    # build Dirichlet vector to apply lift
    Wd = Bcube.assemble_dirichlet_vector(U, V, mesh)

    # Apply lift
    L = L - A * Wd

    # Apply homogeneous dirichlet condition
    Bcube.apply_homogeneous_dirichlet_to_vector!(L, U, V, mesh)
    Bcube.apply_dirichlet_to_matrix!(A, U, V, mesh)

    # Solve linear system
    correction = solve(A, L)
    sol = correction

    sol = sol .+ Wd

    if export_vtk
        velocity, pressure = FEFunction(V, sol)
        vars = Dict("Velocity" => velocity, "Pressure" => pressure)
        write_file(joinpath(outputpath, "output_steady.pvd"), mesh, vars)
    end
    boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
    indices, _ = Bcube.dirichlet_data(U, V, mesh, boundaries)
    return (;
        solution = sol,
        correction,
        matrix = A,
        rhs = L,
        prescribed = Wd,
        indices,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_steady(; export_vtk = get(ENV, "STOKES_EXPORT_VTK", "true") == "true")
end

end #hide
