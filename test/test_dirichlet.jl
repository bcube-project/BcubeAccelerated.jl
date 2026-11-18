using Test, Bcube, BcubeAccelerated, Adapt, KernelAbstractions, SparseArrays, LinearAlgebra

@testset "Dirichlet CPU multithread" begin
    @info "Dirichlet CPU test threads" count=Threads.nthreads()
    mesh_cpu = rectangle_mesh(5, 4)
    scalar = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh_cpu,
        Dict(
            "xmin" => PhysicalFunction(x -> 2 + x[1] + 3*x[2]),
            "ymin" => PhysicalFunction(x -> 2 + x[1] + 3*x[2]),
        ),
    )
    velocity = TrialFESpace(
        FunctionSpace(:Lagrange, 2),
        mesh_cpu,
        Dict(
            "xmin" => PhysicalFunction(x -> [3 + x[1], 4 + 2*x[2]]),
            "ymin" => PhysicalFunction(x -> [3 + x[1], 4 + 2*x[2]]),
        );
        size = 2,
    )
    backend = BcubeAccelerated.BcubeBackendAcc(CPU())
    mesh = adapt(backend, mesh_cpu)
    for U_cpu in (scalar, MultiFESpace(velocity, scalar))
        V_cpu = if U_cpu isa Bcube.AbstractMultiFESpace
            MultiFESpace(map(TestFESpace, Tuple(U_cpu))...)
        else
            TestFESpace(U_cpu)
        end
        U, V = adapt(backend, U_cpu), adapt(backend, V_cpu)
        boundaries_cpu = Bcube.prepare_dirichlet_boundaries(U_cpu, mesh_cpu)
        boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
        expected_data = Bcube.dirichlet_data(U_cpu, V_cpu, mesh_cpu, boundaries_cpu)
        indices, positions = Bcube.dirichlet_data(U, V, mesh, boundaries)
        @test (indices, positions) == expected_data
        @test Bcube.dirichlet_data(U, V, backend, boundaries) == expected_data
        compact_values = Bcube.allocate_dirichlet_values(indices, Float64)
        @test length(compact_values) == length(indices)
        Bcube.evaluate_dirichlet_values!(compact_values, positions, U, V, boundaries, backend, 0.0)
        @test compact_values ==
              Bcube.assemble_dirichlet_vector(U_cpu, V_cpu, mesh_cpu)[indices]
        n = Bcube.get_ndofs(U)
        unconstrained = setdiff(1:n, indices)
        partial_sparse = sparsevec(unconstrained, fill(7.0, length(unconstrained)), n)
        expected_values = Bcube.assemble_dirichlet_vector(U_cpu, V_cpu, mesh_cpu)
        for repetition in 1:3
            @test Bcube.assemble_dirichlet_vector(U, V, mesh) == expected_values
            for reference in
                (fill(7.0, n), spzeros(n), partial_sparse, sparsevec(1:n, fill(7.0, n), n))
                actual, expected = copy(reference), copy(reference)
                Bcube.apply_dirichlet_to_vector!(expected, U_cpu, V_cpu, mesh_cpu)
                Bcube.apply_dirichlet_to_vector!(actual, U, V, mesh)
                @test actual == expected
                Bcube.apply_homogeneous_dirichlet_to_vector!(
                    expected,
                    U_cpu,
                    V_cpu,
                    mesh_cpu,
                )
                Bcube.apply_homogeneous_dirichlet_to_vector!(actual, U, V, mesh)
                @test actual == expected
            end
            for reference in
                (spdiagm(1 => ones(n-1)), spdiagm(0 => ones(n), 1 => ones(n-1)))
                for apply! in (
                    Bcube.apply_dirichlet_to_matrix!,
                    Bcube.apply_homogeneous_dirichlet_to_matrix!,
                )
                    actual = (copy(reference), copy(reference))
                    expected = (copy(reference), copy(reference))
                    apply!(actual, U, V, mesh; diag_values = (2.0, 0.0))
                    apply!(expected, U_cpu, V_cpu, mesh_cpu; diag_values = (2.0, 0.0))
                    @test actual == expected
                end
            end
        end
    end
    U_cpu = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh_cpu)
    U = adapt(backend, U_cpu)
    V = TestFESpace(U)
    n = Bcube.get_ndofs(U)
    b, A = ones(n), spdiagm(0 => ones(n))
    Bcube.apply_dirichlet_to_vector!(b, U, V, mesh)
    Bcube.apply_homogeneous_dirichlet!(A, b, U, V, mesh)
    @test b == ones(n)
    @test A == spdiagm(0 => ones(n))
    @test Bcube.assemble_dirichlet_vector(U, V, mesh) == zeros(n)
end

@testset "Unsorted compact positions on accelerated CPU" begin
    backend = BcubeAccelerated.BcubeBackendAcc(CPU())
    I_diri = [6, 2, 5]
    position_by_dof = zeros(Int, 8)
    position_by_dof[I_diri] = 1:3
    values = [60.0, 20.0, 50.0]
    b = sparsevec([1, 3, 5], [7.0, 8.0, 9.0], 8)
    positions = Bcube.prepare_dirichlet_vector!(b, I_diri, position_by_dof, backend)
    Bcube.set_dirichlet_values!(b, positions, values, backend)
    @test b == [7.0, 20.0, 8.0, 0.0, 50.0, 60.0, 0.0, 0.0]
    @test SparseArrays.nonzeroinds(b)[positions] == I_diri
end

@testset "Typed sparse and dense allocation on the execution backend" begin
    backend = BcubeAccelerated.BcubeBackendAcc(CPU())
    mesh_cpu = rectangle_mesh(3, 2)
    U_cpu = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh_cpu, Dict("xmin" => 1.5))
    V_cpu = TestFESpace(U_cpu)
    mesh, U, V = adapt(backend, mesh_cpu), adapt(backend, U_cpu), adapt(backend, V_cpu)
    for T in (Float32, Float64)
        result = Bcube.assemble_dirichlet_sparse_vector(U, V, mesh; T)
        @test eltype(result) == T
        @test result == Bcube.assemble_dirichlet_sparse_vector(U_cpu, V_cpu, mesh_cpu; T)
        dense_result = Bcube.assemble_dirichlet_vector(U, V, mesh; T)
        @test dense_result isa Vector{T}
        @test dense_result == Vector(result)
    end
end
