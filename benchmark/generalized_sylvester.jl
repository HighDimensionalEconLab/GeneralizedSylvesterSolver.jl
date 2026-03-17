using BenchmarkTools
using GeneralizedSylvesterSolver
using LinearAlgebra
using Random

const GS_SUITE = BenchmarkGroup()

# ── Small square: 2×2 matrices, orders 1–3 ───────────────────────────────────
# Mirrors the 2×2 Schur blocks used in the unit tests (sreal/scplx/treal/tcplx).

for order in (1, 2, 3)
    GS_SUITE["2x2_order$(order)"] = @benchmarkable begin
        generalized_sylvester_solver!(a, b, c, d, $order, ws)
    end setup = begin
        Random.seed!(42)
        a = randn(2, 2)
        b = randn(2, 2)
        c = randn(2, 2)
        d = randn(2, 2^$order)
        ws = GeneralizedSylvesterWs(2, 2, 2, $order)
    end evals = 1
end

# ── Medium square: 4×4 matrices, orders 1–3 ──────────────────────────────────
# Mirrors the 4×4 block Schur matrices used in the unit tests.

for order in (1, 2, 3)
    GS_SUITE["4x4_order$(order)"] = @benchmarkable begin
        generalized_sylvester_solver!(a, b, c, d, $order, ws)
    end setup = begin
        Random.seed!(42)
        a = randn(4, 4)
        b = randn(4, 4)
        c = randn(4, 4)
        d = randn(4, 4^$order)
        ws = GeneralizedSylvesterWs(4, 4, 4, $order)
    end evals = 1
end

# ── Asymmetric: 4×3 matrices (from the end-to-end test in the test suite) ────
# Exercises the case where a/b and c have different sizes.

for order in (1, 2)
    GS_SUITE["4x3_order$(order)"] = @benchmarkable begin
        generalized_sylvester_solver!(a, b, c, d, $order, ws)
    end setup = begin
        Random.seed!(42)
        a = randn(4, 4)
        b = randn(4, 4)
        c = randn(3, 3)
        d = randn(4, 3^$order)
        ws = GeneralizedSylvesterWs(4, 4, 3, $order)
    end evals = 1
end

# ── Workspace-allocation cost: 4×4 order 2, with and without pre-allocation ──
# "alloc_ws" includes GeneralizedSylvesterWs construction in the timed region.
# "solve_only" pre-allocates the workspace in setup so only the solve is timed.

GS_SUITE["4x4_order2_alloc_ws"] = @benchmarkable begin
    ws = GeneralizedSylvesterWs(4, 4, 4, 2)
    generalized_sylvester_solver!(a, b, c, d, 2, ws)
end setup = begin
    Random.seed!(42)
    a = randn(4, 4)
    b = randn(4, 4)
    c = randn(4, 4)
    d = randn(4, 16)
end evals = 1

GS_SUITE["4x4_order2_solve_only"] = @benchmarkable begin
    generalized_sylvester_solver!(a, b, c, d, 2, ws)
end setup = begin
    Random.seed!(42)
    a = randn(4, 4)
    b = randn(4, 4)
    c = randn(4, 4)
    d = randn(4, 16)
    ws = GeneralizedSylvesterWs(4, 4, 4, 2)
end evals = 1

# ── Transpose API: factorize_transpose! and solve_transpose! ─────────────────
# Models the AD (VJP) use case: primal factorize! once, then solve_transpose!
# with a different RHS reusing all cached decompositions.
# "factorize_t_only" times only the transpose Schur phase (reuses LU of A).
# "solve_t_only" times only the back-substitution with pre-built transpose factors.
# "factorize_and_solve_t" times the full transpose phase seen in a VJP.

GS_SUITE["4x4_order2_factorize_t_only"] = @benchmarkable begin
    generalized_sylvester_factorize_transpose!(b, c, 2, ws)
end setup = begin
    Random.seed!(42)
    a = randn(4, 4)
    b = randn(4, 4)
    c = randn(4, 4)
    ws = GeneralizedSylvesterWs(4, 4, 4, 2)
    generalized_sylvester_factorize!(a, b, c, 2, ws)
end evals = 1

GS_SUITE["4x4_order2_solve_t_only"] = @benchmarkable begin
    generalized_sylvester_solve_transpose!(ybar, 2, ws)
end setup = begin
    Random.seed!(42)
    a = randn(4, 4)
    b = randn(4, 4)
    c = randn(4, 4)
    ybar = randn(4, 16)
    ws = GeneralizedSylvesterWs(4, 4, 4, 2)
    generalized_sylvester_factorize!(a, b, c, 2, ws)
    generalized_sylvester_factorize_transpose!(b, c, 2, ws)
end evals = 1

GS_SUITE["4x4_order2_factorize_and_solve_t"] = @benchmarkable begin
    generalized_sylvester_factorize_transpose!(b, c, 2, ws)
    generalized_sylvester_solve_transpose!(ybar, 2, ws)
end setup = begin
    Random.seed!(42)
    a = randn(4, 4)
    b = randn(4, 4)
    c = randn(4, 4)
    ybar = randn(4, 16)
    ws = GeneralizedSylvesterWs(4, 4, 4, 2)
    generalized_sylvester_factorize!(a, b, c, 2, ws)
end evals = 1

# ── Large: 10×10 matrices, orders 1–3 ────────────────────────────────────────
# Matches the matrix size used in PolynomialMatrixEquations.jl tests (the
# closest upstream package in the DynareJulia org that exercises this scale).

for order in (1, 2, 3)
    GS_SUITE["10x10_order$(order)"] = @benchmarkable begin
        generalized_sylvester_solver!(a, b, c, d, $order, ws)
    end setup = begin
        Random.seed!(42)
        a = randn(10, 10)
        b = randn(10, 10)
        c = randn(10, 10)
        d = randn(10, 10^$order)
        ws = GeneralizedSylvesterWs(10, 10, 10, $order)
    end evals = 1
end

# ── Large: 20×20 matrices, orders 1–2 ────────────────────────────────────────
# Representative of medium DSGE models (e.g. fs2000/ls2003 ~12 variables, or
# a small open-economy model with ~20 endogenous variables).

for order in (1, 2)
    GS_SUITE["20x20_order$(order)"] = @benchmarkable begin
        generalized_sylvester_solver!(a, b, c, d, $order, ws)
    end setup = begin
        Random.seed!(42)
        a = randn(20, 20)
        b = randn(20, 20)
        c = randn(20, 20)
        d = randn(20, 20^$order)
        ws = GeneralizedSylvesterWs(20, 20, 20, $order)
    end evals = 1
end

# ── Large: 40×40 matrices, order 2 ───────────────────────────────────────────
# Representative of a large DSGE model (e.g. IRBC with ~20 countries gives
# ~41 endogenous variables). Second-order perturbation (order=2) is the
# primary practical use case at this scale.

GS_SUITE["40x40_order2"] = @benchmarkable begin
    generalized_sylvester_solver!(a, b, c, d, 2, ws)
end setup = begin
    Random.seed!(42)
    a = randn(40, 40)
    b = randn(40, 40)
    c = randn(40, 40)
    d = randn(40, 1600)
    ws = GeneralizedSylvesterWs(40, 40, 40, 2)
end evals = 1

GS_SUITE
