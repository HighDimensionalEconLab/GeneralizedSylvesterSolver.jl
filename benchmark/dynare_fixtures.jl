# Sylvester solver matrices extracted from KOrderPerturbations.jl using the
# Burnside (1998) asset pricing model (Burnside, "Solving asset pricing models
# with Gaussian shocks", JEDC 22, 1998, pp. 329-340).
#
# These use the Dynare/KOrderPerturbations timing convention, where the Kronecker
# factor c = g[1][state_index, 1:nstate] contains only the *predetermined*
# (backward-looking) variables -- a strict subset of all state variables.
# This is structurally different from the DifferentiablePerturbation convention
# used in dsge_fixtures.jl (where c covers ALL state variables).
#
# Model: nvar=2 (y, x), nstate=1 (only x is predetermined), nshock=1, order=2
#   Equation solved: a*X + b*X*kron(c,c) = d
#   Dimensions: a (2×2), b (2×2), c (1×1), d (2×1)
#
# NOTE: These are intentionally small matrices. Their purpose is structural
# regression coverage of the KOrderPerturbations/Dynare timing convention
# (nstate < total state dimension), not performance benchmarking.
#
# To regenerate: julia --project=KOrderPerturbations.jl /tmp/extract_korder_fixtures.jl

const BURN1998_A_SYL = reshape([
    # col 1
    1.0, 0.0,
    # col 2
    -2.273075262432469, 1.0
], 2, 2)

const BURN1998_B_SYL = reshape([
    # col 1
    -0.924831893828356, 0.0,
    # col 2
    18.455271941730015, 0.0
], 2, 2)

const BURN1998_C_SYL = reshape([
    -0.139
], 1, 1)

const BURN1998_D_SYL = reshape([
    0.007979783997977, -0.0
], 2, 1)
