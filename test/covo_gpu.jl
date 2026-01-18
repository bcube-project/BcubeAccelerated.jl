module CovoGpu #hide
println("Running covo example...") #hide

const dir = string(@__DIR__, "/")
using Bcube
using BcubeAccelerated
using KernelAbstractions
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using BcubeGmsh
using Adapt
using BcubeVTK
using LinearAlgebra
using StaticArrays
using StaticArrays
using BenchmarkTools
using UnPack

function compute_residual(_u, V, params, cache)
    u = get_fe_functions(_u)

    # alias on measures
    @unpack dΩ, dΓ, dΓ_perio_x, dΓ_perio_y = params

    # face normals for each face domain (lazy, no computation at this step)
    nΓ = get_face_normals(dΓ)
    nΓ_perio_x = get_face_normals(dΓ_perio_x)
    nΓ_perio_y = get_face_normals(dΓ_perio_y)

    # flux residuals from faces for all variables
    function l(v)
        ∫(flux_Ω(u, v))dΩ +
        -∫(flux_Γ(u, v, nΓ))dΓ +
        -∫(flux_Γ(u, v, nΓ_perio_x))dΓ_perio_x +
        -∫(flux_Γ(u, v, nΓ_perio_y))dΓ_perio_y
    end
    # l1(v) = ∫(flux_Ω(u, v))dΩ
    # l2(v) = ∫(flux_Γ(u, v, nΓ))dΓ
    # l3(v) = ∫(flux_Γ(u, v, nΓ_perio_x))dΓ_perio_x
    # l4(v) = ∫(flux_Γ(u, v, nΓ_perio_y))dΓ_perio_y
    # @show "l1"
    # assemble_linear(l1, V)
    # @show "l2"
    # assemble_linear(l2, V)
    # @show "l3"
    # assemble_linear(l3, V)
    # @show "l4"
    # assemble_linear(l4, V)
    # error("ici")
    _rhs = assemble_linear(l, V)
    rhs = similar(_rhs)
    Bcube.__solve!(rhs, cache.mass, _rhs, Bcube.get_bcube_backend(dΩ))
    return rhs
end

"""
    flux_Ω(u, v)

Compute volume residual using the lazy-operators approach
"""
flux_Ω(u, v) = _flux_Ω ∘ cellvar(u, v)
cellvar(u, v) = (u, map(∇, v))
function _flux_Ω(u, ∇v)
    ρ, ρu, ρE, ϕ = u
    ∇λ_ρ, ∇λ_ρu, ∇λ_ρE, ∇λ_ϕ = ∇v

    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p   = pressure(ρ, ρu, ρE, γ)

    flux_ρ  = ρu
    flux_ρu = ρuu + p * I
    flux_ρE = (ρE + p) .* vel
    flux_ϕ  = ϕ .* vel

    return ∇λ_ρ ⋅ flux_ρ + ∇λ_ρu ⊡ flux_ρu + ∇λ_ρE ⋅ flux_ρE + ∇λ_ϕ ⋅ flux_ϕ
end

"""
    flux_Γ(u,v,n)

Flux at the interface is defined by a composition of two functions:
* facevar(u,v,n) defines the input states which are needed for
  the riemann flux using operator notations
* flux_roe(w) defines the Riemann flux (as usual)
"""
flux_Γ(u, v, n) = flux_roe ∘ (side⁻(u), side⁺(u), jump(v), side⁻(n))

"""
    flux_roe(w)
"""
function flux_roe(ui, uj, δv, nij)
    # destructuring inputs given by `facevar` function

    nx, ny = nij
    ρ1, ρu1, ρE1, ϕ1 = ui
    ρ2, ρu2, ρE2, ϕ2 = uj
    δλ_ρ1, δλ_ρu1, δλ_ρE1, δλ_ϕ1 = δv
    ρux1, ρuy1 = ρu1
    ρux2, ρuy2 = ρu2

    # Closure
    u1 = ρux1 / ρ1
    v1 = ρuy1 / ρ1
    u2 = ρux2 / ρ2
    v2 = ρuy2 / ρ2
    p1 = pressure(ρ1, ρu1, ρE1, γ)
    p2 = pressure(ρ2, ρu2, ρE2, γ)

    H2 = (γ / (γ - 1)) * p2 / ρ2 + (u2 * u2 + v2 * v2) / 2.0
    H1 = (γ / (γ - 1)) * p1 / ρ1 + (u1 * u1 + v1 * v1) / 2.0

    R = √(ρ1 / ρ2)
    invR1 = 1.0 / (R + 1)
    uAv = (R * u1 + u2) * invR1
    vAv = (R * v1 + v2) * invR1
    Hav = (R * H1 + H2) * invR1
    cAv = √(abs((γ - 1) * (Hav - (uAv * uAv + vAv * vAv) / 2.0)))
    ecAv = (uAv * uAv + vAv * vAv) / 2.0

    λ1 = nx * uAv + ny * vAv
    λ3 = λ1 + cAv
    λ4 = λ1 - cAv

    d1 = ρ1 - ρ2
    d2 = ρ1 * u1 - ρ2 * u2
    d3 = ρ1 * v1 - ρ2 * v2
    d4 = ρE1 - ρE2

    # computation of the centered part of the flux
    flu11 = nx * ρ2 * u2 + ny * ρ2 * v2
    flu21 = nx * p2 + flu11 * u2
    flu31 = ny * p2 + flu11 * v2
    flu41 = H2 * flu11

    # Temp variables
    rc1 = (γ - 1) / cAv
    rc2 = (γ - 1) / cAv / cAv
    uq41 = ecAv / cAv + cAv / (γ - 1)
    uq42 = nx * uAv + ny * vAv

    fdc1 = max(λ1, 0.0) * (d1 + rc2 * (-ecAv * d1 + uAv * d2 + vAv * d3 - d4))
    fdc2 = max(λ1, 0.0) * ((nx * vAv - ny * uAv) * d1 + ny * d2 - nx * d3)
    fdc3 =
        max(λ3, 0.0) * (
            (-uq42 * d1 + nx * d2 + ny * d3) / 2.0 +
            rc1 * (ecAv * d1 - uAv * d2 - vAv * d3 + d4) / 2.0
        )
    fdc4 =
        max(λ4, 0.0) * (
            (uq42 * d1 - nx * d2 - ny * d3) / 2.0 +
            rc1 * (ecAv * d1 - uAv * d2 - vAv * d3 + d4) / 2.0
        )

    duv1 = fdc1 + (fdc3 + fdc4) / cAv
    duv2 = uAv * fdc1 + ny * fdc2 + (uAv / cAv + nx) * fdc3 + (uAv / cAv - nx) * fdc4
    duv3 = vAv * fdc1 - nx * fdc2 + (vAv / cAv + ny) * fdc3 + (vAv / cAv - ny) * fdc4
    duv4 =
        ecAv * fdc1 +
        (ny * uAv - nx * vAv) * fdc2 +
        (uq41 + uq42) * fdc3 +
        (uq41 - uq42) * fdc4

    v₁₂ = 0.5 * ((u1 + u2) * nx + (v1 + v2) * ny)
    fluxϕ = max(0.0, v₁₂) * ϕ1 + min(0.0, v₁₂) * ϕ2

    return (
        δλ_ρ1 * (flu11 + duv1) +
        δλ_ρu1 ⋅ (SA[flu21 + duv2, flu31 + duv3]) +
        δλ_ρE1 * (flu41 + duv4) +
        δλ_ϕ1 * (fluxϕ)
    )
end

"""
Time integration of `f(q, t)` over a timestep `Δt`.
"""
forward_euler(q, f::Function, t, Δt) = get_dof_values(q) .+ Δt .* f(q, t)

"""
    rk3_ssp(q, f::Function, t, Δt)

`f(q, t)` is the function to integrate.
"""
function rk3_ssp(q, f::Function, t, Δt)
    stepper(q, t) = forward_euler(q, f, t, Δt)
    _q0 = get_dof_values(q)

    _q1 = stepper(q, Δt)

    set_dof_values!(q, _q1)
    _q2 = (3 / 4) .* _q0 .+ (1 / 4) .* stepper(q, t + Δt)

    set_dof_values!(q, _q2)
    _q1 .= (1 / 3) * _q0 .+ (2 / 3) .* stepper(q, t + Δt / 2)

    return _q1
end

"""
    pressure(ρ, ρu, ρE, γ)

Computes pressure from perfect gaz law
"""
function pressure(ρ::Number, ρu::AbstractVector, ρE::Number, γ)
    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = (γ - 1) * (ρE - tr(ρuu) / 2)
    return p
end

"""
  Init field with a vortex (for the COVO test case)
"""
function covo!(q, dΩ)

    # Intermediate vars
    Cₚ = γ * r / (γ - 1)

    r²(x) = ((x[1] .- xvc) .^ 2 + (x[2] .- yvc) .^ 2) ./ Rc^2
    # Temperature
    T(x) = T₀ .- β^2 * U₀^2 / (2 * Cₚ) .* exp.(-r²(x))
    # Velocity
    ux(x) = U₀ .- β * U₀ / Rc .* (x[2] .- yvc) .* exp.(-r²(x) ./ 2)
    uy(x) = V₀ .+ β * U₀ / Rc .* (x[1] .- xvc) .* exp.(-r²(x) ./ 2)
    # Density
    ρ(x) = ρ₀ .* (T(x) ./ T₀) .^ (1.0 / (γ - 1))
    # momentum
    ρu(x) = SA[ρ(x) * ux(x), ρ(x) * uy(x)]
    # Energy
    ρE(x) = ρ(x) * ((Cₚ / γ) .* T(x) + (ux(x) .^ 2 + uy(x) .^ 2) ./ 2)
    # Passive scalar
    ϕ(x) = Rc^2 * r²(x) < 0.01 ? exp(-r²(x) ./ 2) : 0.0

    f = map(PhysicalFunction, (ρ, ρu, ρE, ϕ))
    projection_l2!(q, f, dΩ)
    return nothing
end

"""
    Tiny struct to ease the VTK output
"""
mutable struct VtkHandler
    basename::Any
    mesh::Any
    ite::Any
    VtkHandler(basename, mesh) = new(basename, mesh, 0)
end

"""
    Write solution to vtk
    Wrapper for `write_vtk`
"""
function append_vtk(vtk, vars, t, params)
    ρ, ρu, ρE, ϕ = vars

    mesh = vtk.mesh
    mesh_degree = 1
    vtk_degree = maximum(x -> get_degree(Bcube.get_function_space(get_fespace(x))), vars)
    vtk_degree = max(1, mesh_degree, vtk_degree)

    # p = pressure(ρ, ρu, ρE, γ)
    p = pressure ∘ (ρ, ρu, ρE, γ)
    dict_vars_dg = Dict("rho" => ρ, "rhou" => ρu, "rhoE" => ρE, "phi" => ϕ, "p" => p)
    write_file(
        vtk.basename * "_DG.pvd",
        mesh,
        dict_vars_dg,
        vtk.ite,
        t;
        discontinuous = true,
        mesh_degree = vtk_degree,
        collection_append = vtk.ite > 0,
    )

    # Update counter
    vtk.ite += 1
end

function Adapt.adapt_structure(to, mesh::BcubeGmsh.GmshMetaData)
    nothing
end
function Bcube.__solve!(
    x,
    A,
    b,
    backend::BcubeAccelerated.BcubeBackendAcc{K, CUDA.CUDABackend},
) where {K}
    F = CUSOLVER.SparseCholesky(A)
    CUSOLVER.spcholesky_factorise(F, A, 1.e-12)
    CUSOLVER.spcholesky_solve(F, b, x)
    return nothing
end
function Bcube.__solve!(
    x,
    A::CUSOLVER.SparseCholesky,
    b,
    backend::BcubeAccelerated.BcubeBackendAcc{K, CUDA.CUDABackend},
) where {K}
    CUSOLVER.spcholesky_solve(A, b, x)
    return nothing
end
# function Bcube.get_bcube_backend(a::CuArray)
#     BcubeAccelerated.BcubeBackendAcc(KernelAbstractions.get_backend(a))
# end

# Settings
if get(ENV, "BenchmarkMode", "false") == "false" #hide
    const cellfactor = 1
    const nx = 32 * cellfactor + 1
    const ny = 32 * cellfactor + 1
    const fspace = :Lagrange
    const timeScheme = :ForwardEuler
else #hide
    const nx = 128 + 1 #hide
    const ny = 128 + 1 #hide
    const fspace = :Lagrange
    const timeScheme = :ForwardEuler
end #hide
const nperiod = 1 # number of turn
const CFL = 0.1
const degree = 1 # FunctionSpace degree
const degquad = 2 * degree + 1
const γ = 1.4
const β = 0.2 # vortex intensity
const r = 287.15 # Perfect gaz constant
const T₀ = 300 # mean-flow temperature
const P₀ = 1e5 # mean-flow pressure
const M₀ = 0.5 # mean-flow mach number
const ρ₀ = 1.0 # mean-flow density
const xvc = 0.0 # x-center of vortex
const yvc = 0.0 # y-center of vortex
const Rc = 0.005 # Charasteristic vortex radius
const c₀ = √(γ * r * T₀) # Sound velocity
const U₀ = M₀ * c₀ # mean-flow velocity
const V₀ = 0.0 # mean-flow velocity
const ϕ₀ = 1.0
const l = 0.05 # half-width of the domain
const Δt = CFL * 2 * l / (nx - 1) / ((1 + β) * U₀ + c₀) / (2 * degree + 1)
#const Δt = 5.e-7
const nout = 100 # Number of time steps to save
const outputpath = "./covo/"
const output = joinpath(@__DIR__, outputpath, "covo_deg$degree")
const nite = Int(floor(nperiod * 2 * l / (U₀ * Δt))) + 1

function build_mass_matrix_test(U, V, dΩ::D) where {D <: AbstractMeasure}
    _m(u, v) = ∫(u ⋅ v)dΩ
    # @code_warntype  _m(1,2)
    @show typeof(_m(1, 2))
    BcubeAccelerated.test_arg_AK_foreachindex(get_backend(Bcube.get_bcube_backend(dΩ)), dΩ)
    BcubeAccelerated.test_arg_AK_foreachindex(
        get_backend(Bcube.get_bcube_backend(dΩ)),
        _m(1, 2),
    )
    return assemble_bilinear(_m, U, V)
end
function setup(backend, tmp_path, fs)
    mesh_cpu = read_mesh(tmp_path)
    rm(tmp_path)
    U_sca_cpu = TrialFESpace(fs, mesh_cpu, :discontinuous; size = 1) # DG, scalar
    U_vec_cpu = TrialFESpace(fs, mesh_cpu, :discontinuous; size = 2) # DG, vectoriel
    U_cpu = MultiFESpace(U_sca_cpu, U_vec_cpu, U_sca_cpu, U_sca_cpu)
    V_sca_cpu = TestFESpace(U_sca_cpu)
    V_vec_cpu = TestFESpace(U_vec_cpu)
    V_cpu = MultiFESpace(V_sca_cpu, V_vec_cpu, V_sca_cpu, V_sca_cpu)

    mesh = adapt(backend, mesh_cpu)
    BcubeAccelerated.test_arg(backend, mesh)
    U = adapt(backend, U_cpu)
    V = adapt(backend, V_cpu)
    BcubeAccelerated.test_arg(backend, U)
    BcubeAccelerated.test_arg(backend, V)
    return mesh, U, V, mesh_cpu
end

_factorize(backend::Bcube.AbstractBcubeBackend, M) = __factorize(get_backend(backend), M)
__factorize(backend, M) = factorize(M)
function __factorize(backend::CUDA.CUDABackend, M)
    F = CUSOLVER.SparseCholesky(M)
    CUSOLVER.spcholesky_factorise(F, M, 1.e-12)
    return F
end

function run_covo(backend)
    println("Starting run_covo...")

    # Build mesh
    meshParam = (nx = nx, ny = ny, lx = 2l, ly = 2l, xc = 0.0, yc = 0.0)
    tmp_path = "tmp.msh"
    if get(ENV, "BenchmarkMode", "false") == "false" #hide
        BcubeGmsh.gen_rectangle_mesh(tmp_path, :quad; meshParam...)
    else #hide
        if get(ENV, "MeshConfig", "quad") == "triquad" #hide
            BcubeGmsh.gen_rectangle_mesh_with_tri_and_quad(tmp_path; meshParam...) #hide
        else #hide
            BcubeGmsh.gen_rectangle_mesh(tmp_path, :quad; meshParam...) #hide
        end #hide
    end #hide

    # Define variables and test functions
    fs = FunctionSpace(fspace, degree)
    mesh, U, V, mesh_cpu = setup(backend, tmp_path, fs)
    u = FEFunction(U, KernelAbstractions.zeros(backend, Float64, get_ndofs(U)))

    @show Bcube.get_ndofs(U)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)

    # Declare periodic boundary conditions and
    # create associated domains and measures
    periodicBCType_x = PeriodicBCType(Translation(SA[-2l, 0.0]), ("East",), ("West",))
    periodicBCType_y = PeriodicBCType(Translation(SA[0.0, 2l]), ("South",), ("North",))
    Γ_perio_x_cpu = BoundaryFaceDomain(mesh_cpu, periodicBCType_x)
    Γ_perio_y_cpu = BoundaryFaceDomain(mesh_cpu, periodicBCType_y)
    Γ_perio_x = adapt(backend, Γ_perio_x_cpu)
    Γ_perio_y = adapt(backend, Γ_perio_y_cpu)
    dΓ_perio_x = Measure(Γ_perio_x, degquad)
    dΓ_perio_y = Measure(Γ_perio_y, degquad)

    params = (dΩ = dΩ, dΓ = dΓ, dΓ_perio_x = dΓ_perio_x, dΓ_perio_y = dΓ_perio_y)

    # Init vtk
    mkpath(joinpath(@__DIR__, outputpath))
    vtk = VtkHandler(output, mesh_cpu)

    # Init solution
    t = 0.0

    covo!(u, dΩ)

    # cache mass matrices
    cache = (mass = _factorize(backend, Bcube.build_mass_matrix(U, V, dΩ)),)

    if get(ENV, "BenchmarkMode", "false") == "true" #hide
        return u, U, V, params, cache
    end

    # Write initial solution
    append_vtk(vtk, map(x -> adapt(get_backend(zeros(1)), x), (u...,)), t, params)

    # Time loop
    for i in 1:nite
        println("")
        println("")
        println("Iteration ", i, " / ", nite)

        ## Step forward in time
        rhs(u, t) = compute_residual(u, V, params, cache)
        if timeScheme == :ForwardEuler
            unew = forward_euler(u, rhs, t, Δt)
        elseif timeScheme == :RK3
            unew = rk3_ssp(u, rhs, t, Δt)
        else
            error("Unknown time scheme: $timeScheme")
        end

        set_dof_values!(u, unew)

        t += Δt

        # Write solution to file
        if (i % Int(max(floor(nite / nout), 1)) == 0)
            println("--> VTK export")
            append_vtk(vtk, map(x -> adapt(get_backend(zeros(1)), x), (u...,)), t, params)
        end
    end

    # Summary and benchmark                                 # ndofs total = 20480
    _rhs(u, t) = compute_residual(u, V, params, cache)
    @btime begin
        forward_euler($u, $_rhs, $time, $Δt)  # 5.639 ms (1574 allocations: 2.08 MiB)
        KernelAbstractions.synchronize(get_backend($backend))
    end
    println("ndofs total = ", Bcube.get_ndofs(U))
    @show Δt, U₀, U₀ * t
    @show boundary_names(mesh)
    return nothing
end

#const backendDevice = get_backend(ones(2))  ## CPU
const backendDevice = get_backend(CUDA.ones(2)) ## GPU
const backend = BcubeAccelerated.BcubeBackendAcc(
    backendDevice;
    kernel = BcubeAccelerated.KernelKA(),
    sync = true,
)

#const backend = Bcube.get_bcube_backend()
@show backend

if get(ENV, "BenchmarkMode", "false") == "false"
    mkpath(outputpath)
    run_covo(backend)
end

end #hide

# CPU serial
# 330.144 ms (52 allocations: 31.00 MiB)
# ndofs total = 327680

# CPU (1threads)  nx=128
# 336.991 ms (397 allocations: 31.03 MiB)
# ndofs total = 327680

# CPU (80threads)  nx=128
# 21.236 ms (2741 allocations: 32.05 MiB)
# ndofs total = 327680

# GPU nx=128
# 8.990 ms (13675 allocations: 1.22 MiB)
# ndofs total = 327680