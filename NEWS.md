0.2.2
=====
- Cache LU(A) and Schur decompositions in `GeneralizedSylvesterWs`; expose
  `generalized_sylvester_factorize!` / `generalized_sylvester_solve!` so the
  factorization can be reused across solves.
- Add `generalized_sylvester_factorize_transpose!` /
  `generalized_sylvester_solve_transpose!` for the transposed system
  `A'·P + B'·P·(C'⊗...⊗C') = Y`, reusing the cached `LU(A)` (useful for
  reverse-mode/VJP applications).
- Add an opt-in dense fallback for rank-deficient `A\B`, plus optional Schur
  balancing, controlled by the new `balance::Bool` and `deflate_tol::Float64`
  keyword arguments. Defaults are `balance=false`, `deflate_tol=0.0`, which
  preserves the previous behavior bit-for-bit when the kwargs are omitted.
  Callers needing the new behavior can opt in with
  `generalized_sylvester_solver!(a, b, c, d, order, ws; balance=true, deflate_tol=-1.0)`.
- Move DSGE regression fixtures from `benchmark/dsge_fixtures.jl` to
  `test/dsge_fixtures.jl` so they ship with the test suite.
- Bump `julia` compat to `1.10.7` (the real lower bound implied by
  `FastLapackInterface v2`).

0.2.1
=====
- update to FastLapackInterface v.2.0.0

0.2.0
=====
- fix various issues
- update KroneckerTools

0.1.2
=====
- code refactoring

0.1.0
=====
- initial files
