using Test, LinearAlgebra
include("stokes_flow.jl")
@info "Stokes CPU test threads" count=Threads.nthreads()
@testset "Stokes CPU" begin
    @test Threads.nthreads() == 2
    reference_path = joinpath(@__DIR__, "assets", "stokes_steady_reference.txt")
    reference = [parse(Float64, line) for line in eachline(reference_path) if !startswith(line, '#')]

    for repetition in 1:3
        result = stokes_flow_API.run_steady(; accelerated = true, export_vtk = false)
        relative_difference = norm(result.solution - reference) / norm(reference)
        @show repetition relative_difference
        @test relative_difference < 1e-9
    end
end
