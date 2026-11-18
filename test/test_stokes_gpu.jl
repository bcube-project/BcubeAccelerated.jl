if !isdefined(@__MODULE__, :stokes_flow_API)
    include("stokes_flow.jl")
end

@testset "Stokes CUDA" begin
    CUDA.allowscalar(false)
    reference_path = joinpath(@__DIR__, "assets", "stokes_steady_reference.txt")
    reference = [parse(Float64, line) for line in eachline(reference_path) if !startswith(line, '#')]

    solve = (matrix, rhs) -> CUDA.CUSOLVER.csrlsvqr!(
        CUDA.CUSPARSE.CuSparseMatrixCSR(matrix), rhs, similar(rhs), 1e-12, Cint(0), 'O',
    )
    result = stokes_flow_API.run_steady(;
        backend_device = CUDA.CUDABackend(), solve, export_vtk = false,
    )
    @test result.solution isa CUDA.CuArray
    @test norm(result.matrix * result.correction - result.rhs) / norm(result.rhs) < 1e-9
    @test norm(Array(result.solution) - reference) / norm(reference) < 1e-9
end
