using BenchmarkTools
using GeneralizedSylvesterSolver

const SUITE = BenchmarkGroup()
SUITE["generalized_sylvester"] = include(
    joinpath(pkgdir(GeneralizedSylvesterSolver), "benchmark", "generalized_sylvester.jl"),
)
SUITE["dsge_models"] = include(
    joinpath(pkgdir(GeneralizedSylvesterSolver), "benchmark", "dsge_models.jl"),
)
SUITE["dynare_models"] = include(
    joinpath(pkgdir(GeneralizedSylvesterSolver), "benchmark", "dynare_models.jl"),
)

SUITE
