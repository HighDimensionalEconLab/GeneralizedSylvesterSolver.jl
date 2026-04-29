using BenchmarkTools
using GeneralizedSylvesterSolver

# Real DSGE matrices extracted from DifferentiablePerturbation.jl second-order perturbation.
# See dsge_fixtures.jl for provenance and regeneration instructions.
include(joinpath(pkgdir(GeneralizedSylvesterSolver), "test", "dsge_fixtures.jl"))

const DSGE_SUITE = BenchmarkGroup()

# ── RBC (n=4, n_x=2) ────────────────────────────────────────────────────────
# Small textbook RBC model; matches existing 4×4_order2 size but with real matrices.

DSGE_SUITE["rbc"] = @benchmarkable begin
    generalized_sylvester_solver!(a, b, c, d, 2, ws)
end setup = begin
    a  = copy(RBC_A_SYL)
    b  = copy(RBC_C_SYL)
    c  = copy(RBC_H_X)
    d  = copy(RBC_E_SYL)
    ws = GeneralizedSylvesterWs(4, 4, 2, 2)
end evals = 1

# ── SGU (n=15, n_x=4) ───────────────────────────────────────────────────────
# Schmitt-Grohé & Uribe (2003) small open-economy model.

DSGE_SUITE["sgu"] = @benchmarkable begin
    generalized_sylvester_solver!(a, b, c, d, 2, ws)
end setup = begin
    a  = copy(SGU_A_SYL)
    b  = copy(SGU_C_SYL)
    c  = copy(SGU_H_X)
    d  = copy(SGU_E_SYL)
    ws = GeneralizedSylvesterWs(15, 15, 4, 2)
end evals = 1

# ── FVGQ (n=38, n_x=14) ─────────────────────────────────────────────────────
# Fernández-Villaverde et al. (2010) medium-scale DSGE; largest realistic case.

DSGE_SUITE["fvgq"] = @benchmarkable begin
    generalized_sylvester_solver!(a, b, c, d, 2, ws)
end setup = begin
    a  = copy(FVGQ_A_SYL)
    b  = copy(FVGQ_C_SYL)
    c  = copy(FVGQ_H_X)
    d  = copy(FVGQ_E_SYL)
    ws = GeneralizedSylvesterWs(38, 38, 14, 2)
end evals = 1

DSGE_SUITE
