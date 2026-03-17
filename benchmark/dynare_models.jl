using BenchmarkTools
using GeneralizedSylvesterSolver

include(joinpath(pkgdir(GeneralizedSylvesterSolver), "benchmark", "dynare_fixtures.jl"))

# Benchmarks using matrices from the KOrderPerturbations/Dynare timing convention.
# In this convention c = g[1][state_index, 1:nstate] contains only the
# predetermined (backward-looking) variables, so nstate < total state dimension.
# See dynare_fixtures.jl for details and regeneration instructions.

const DYNARE_SUITE = BenchmarkGroup()

# Burnside (1998): nvar=2, nstate=1, order=2  (2×2 system, 1×1 Kronecker factor)
# Small matrices; this benchmark exists for structural regression coverage of the
# Dynare timing convention, not for performance measurement.
DYNARE_SUITE["burn1998_order2"] = @benchmarkable begin
    generalized_sylvester_solver!(a, b, c, d, 2, ws)
end setup = begin
    a  = copy(BURN1998_A_SYL)
    b  = copy(BURN1998_B_SYL)
    c  = copy(BURN1998_C_SYL)
    d  = copy(BURN1998_D_SYL)
    ws = GeneralizedSylvesterWs(2, 2, 1, 2)
end evals = 1

DYNARE_SUITE
