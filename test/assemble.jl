function eval_assemble(mesh, U_p, U_u)
    V_p = TestFESpace(U_p)
    V_u = TestFESpace(U_u)

    V = MultiFESpace(V_p, V_u)

    dΩ = Measure(CellDomain(mesh), 1)
    l((v_u, v_p)) = ∫(v_p)dΩ

    y = Bcube.allocate_dofs(V)
    assemble_linear!(y, l, V)
    return y
end

@testset "assemble" begin
    # Build ref result, then KA
    mesh_cpu = one_cell_mesh(:line)
    fs = FunctionSpace(:Lagrange, 1)
    U_p_cpu = TrialFESpace(fs, mesh_cpu)
    U_u_cpu = TrialFESpace(fs, mesh_cpu)
    y_cpu = eval_assemble(mesh_cpu, U_p_cpu, U_u_cpu)

    backendDevice = get_backend(ones(2))
    backend = BcubeAccelerated.BcubeBackendAcc(
        backendDevice;
        kernel = BcubeAccelerated.KernelKA(),
        sync = true,
    )
    mesh = adapt(backend, mesh_cpu)
    U_p = adapt(backend, U_p_cpu)
    U_u = adapt(backend, U_u_cpu)
    y_KA_cpu = eval_assemble(mesh, U_p, U_u)
    @test all(y_cpu .= y_KA_cpu)
end