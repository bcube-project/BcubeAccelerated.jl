@testset "Dirichlet Float64 dense CUDA" begin
    mesh_cpu = rectangle_mesh(3, 3)
    amplitude = 3.0
    boundary_value = time -> PhysicalFunction(point ->
        2.0 + point[1] + amplitude * point[2] + time,
    )
    # The two tags share a corner and prescribe the same value there.
    trial_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh_cpu,
        Dict("xmin" => boundary_value, "ymin" => boundary_value),
    )
    test_cpu = TestFESpace(trial_cpu)
    time = 0.75
    expected = fill(7.0, Bcube.get_ndofs(trial_cpu))
    Bcube.apply_dirichlet_to_vector!(expected, trial_cpu, test_cpu, mesh_cpu, time)
    boundaries_cpu = Bcube.prepare_dirichlet_boundaries(trial_cpu, mesh_cpu)
    expected_indices, expected_inverse =
        Bcube.dirichlet_data(trial_cpu, test_cpu, mesh_cpu, boundaries_cpu)

    backend = BcubeAccelerated.BcubeBackendAcc(CUDA.CUDABackend())
    mesh = adapt(backend, mesh_cpu)
    trial = adapt(backend, trial_cpu)
    test = adapt(backend, test_cpu)
    actual = CUDA.CuArray(fill(7.0, length(expected)))

    CUDA.allowscalar(false)
    boundaries = Bcube.prepare_dirichlet_boundaries(trial, mesh)
    indices, inverse = Bcube.dirichlet_data(trial, test, mesh, boundaries)
    @test indices isa CUDA.CuArray
    @test inverse isa CUDA.CuArray
    @test Array(indices) == expected_indices
    @test Array(inverse) == expected_inverse

    Bcube.apply_dirichlet_to_vector!(actual, trial, test, mesh, time)
    @test Array(actual) ≈ expected

    @testset "Sparse storage and assembly" begin
        ndofs = length(expected)
        assembled_sparse = Bcube.assemble_dirichlet_sparse_vector(trial, test, mesh, time)
        reference_sparse =
            Bcube.assemble_dirichlet_sparse_vector(trial_cpu, test_cpu, mesh_cpu, time)
        @test assembled_sparse isa CUDA.CUSPARSE.CuSparseVector
        @test SparseVector(assembled_sparse) ≈ reference_sparse
        assembled_dense = Bcube.assemble_dirichlet_vector(trial, test, mesh, time)
        @test assembled_dense isa CUDA.CuArray
        @test Array(assembled_dense) ≈ Vector(reference_sparse)

        free_dofs = setdiff(1:ndofs, expected_indices)
        for stored_dofs in (1:ndofs, free_dofs, Int[])
            reference = sparsevec(stored_dofs, fill(-0.0, length(stored_dofs)), ndofs)
            vector = CUDA.CUSPARSE.CuSparseVector(reference)
            Bcube.apply_dirichlet_to_vector!(reference, trial_cpu, test_cpu, mesh_cpu, time)
            Bcube.apply_dirichlet_to_vector!(vector, trial, test, mesh, time)
            @test Array(SparseArrays.nonzeroinds(vector)) == SparseArrays.nonzeroinds(reference)
            @test isequal(Array(nonzeros(vector)), nonzeros(reference))
        end

        for diagonal in (1:ndofs, free_dofs), apply! in (
            Bcube.apply_dirichlet_to_matrix!,
            Bcube.apply_homogeneous_dirichlet_to_matrix!,
        )
            reference = sparse(diagonal, diagonal, fill(4.0, length(diagonal)), ndofs, ndofs)
            reference += spdiagm(-1 => ones(ndofs - 1), 1 => ones(ndofs - 1))
            matrix = CUDA.CUSPARSE.CuSparseMatrixCSC(reference)
            apply!(reference, trial_cpu, test_cpu, mesh_cpu; diag_value = 2.0)
            apply!(matrix, trial, test, mesh; diag_value = 2.0)
            actual_csc = SparseMatrixCSC(matrix)
            @test actual_csc.colptr == reference.colptr
            @test rowvals(actual_csc) == rowvals(reference)
            @test nonzeros(actual_csc) == nonzeros(reference)
        end
    end
end
