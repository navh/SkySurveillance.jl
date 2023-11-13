using SkySurveillance
using Test
using Aqua
using JET

@testset "SkySurveillance.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SkySurveillance)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SkySurveillance; target_defined_modules = true)
    end
    # Write your tests here.
end
