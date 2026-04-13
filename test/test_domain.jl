@testset "Domain" begin
    _backend = get_backend(ones(1))
    backend = BcubeAccelerated.BcubeBackendAcc(
        _backend;
        kernel = BcubeAccelerated.KernelKA(),
        sync = true,
    )

    mesh_cpu = rectangle_mesh(2, 3)
    mesh = adapt(backend, mesh_cpu)
    Ω = CellDomain(mesh)

    f1 = PhysicalFunction(x -> x[1])
    @test Bcube.get_return_type_and_codim(f1, Ω, backend) == (Float64, (1,))
    f2 = PhysicalFunction(x -> x)
    @test Bcube.get_return_type_and_codim(f2, Ω, backend) == (Float64, (2,))
end