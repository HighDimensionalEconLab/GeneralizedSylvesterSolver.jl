module GeneralizedSylvesterSolver
###
# solving a x + b x (c ⊗ c ⊗ ... ⊗ c) = d
# using (I + s^T ⊗ s^T ⊗ ... \otimes s^T \otimes t)x = d
###

using QuasiTriangular
using FastLapackInterface
using KroneckerTools
using LinearAlgebra
import LinearAlgebra: ldiv!, mul!

export GeneralizedSylvesterWs,
       generalized_sylvester_solver!,
       generalized_sylvester_factorize!,
       generalized_sylvester_solve!,
       generalized_sylvester_factorize_transpose!,
       generalized_sylvester_solve_transpose!

# Holds pre-computed data for dense fallback when A\B is rank-deficient.
struct RankDeficientData
    n_deflated::Int                                                      # number of zero eigenvalues (moved to top of Schur form)
    T12::Matrix{Float64}                                                 # cross-block: n_deflated × r
    kron_S::Matrix{Float64}                                              # kron(S, S): n_x^order × n_x^order
    dense_lu::LinearAlgebra.LU{Float64, Matrix{Float64}, Vector{Int64}}  # LU of (I + kron(kron(S',S'), T22))
end

mutable struct GeneralizedSylvesterWs
    ma::Int64
    mb::Int64
    b1::Matrix{Float64}
    c1::Matrix{Float64}
    a1::Matrix{Float64}
    s2::QuasiUpperTriangular{Float64,Matrix{Float64}}
    t2::QuasiUpperTriangular{Float64,Matrix{Float64}}
    work1::Vector{Float64}
    work2::Vector{Float64}
    work3::Vector{Float64}
    work4::Vector{Float64}
    result::Matrix{Float64}
    linsolve::LUWs
    schur_b::SchurWs
    schur_c::SchurWs
    # Balancing info (forward system)
    bal_ilo::Base.RefValue{Int}
    bal_ihi::Base.RefValue{Int}
    bal_scale::Vector{Float64}
    # Transpose system
    b1_t::Matrix{Float64}
    c1_t::Matrix{Float64}
    s2_t::QuasiUpperTriangular{Float64,Matrix{Float64}}
    t2_t::QuasiUpperTriangular{Float64,Matrix{Float64}}
    schur_bt::SchurWs
    schur_ct::SchurWs
    # Balancing info (transpose system)
    bal_ilo_t::Base.RefValue{Int}
    bal_ihi_t::Base.RefValue{Int}
    bal_scale_t::Vector{Float64}
    # Rank-deficient dense solve data (nothing when full rank)
    rddata::Union{Nothing, RankDeficientData}
    rddata_t::Union{Nothing, RankDeficientData}
    function GeneralizedSylvesterWs(ma::Int64, mb::Int64, mc::Int64, order::Int64)
        if mb != ma
            DimensionMismatch("a has $ma rows but b has $mb rows")
        end
        b1 = Matrix{Float64}(undef, mb,mb)
        c1 = Matrix{Float64}(undef, mc,mc)
        a1 = Matrix{Float64}(undef, ma,ma)
        s2 = QuasiUpperTriangular(Matrix{Float64}(undef, mc,mc))
        t2 = QuasiUpperTriangular(Matrix{Float64}(undef, mb,mb))
        linsolve = LUWs(ma)
        schur_b = SchurWs(b1)
        schur_c = SchurWs(c1)
        work1 = Vector{Float64}(undef, ma*mc^order)
        work2 = Vector{Float64}(undef, ma*mc^order)
        work3 = Vector{Float64}(undef, ma*mc^order)
        work4 = Vector{Float64}(undef, ma*mc^order)
        result = Matrix{Float64}(undef, ma,mc^order)
        bal_ilo = Ref(1)
        bal_ihi = Ref(ma)
        bal_scale = ones(Float64, ma)
        b1_t = Matrix{Float64}(undef, mb,mb)
        c1_t = Matrix{Float64}(undef, mc,mc)
        s2_t = QuasiUpperTriangular(Matrix{Float64}(undef, mc,mc))
        t2_t = QuasiUpperTriangular(Matrix{Float64}(undef, mb,mb))
        schur_bt = SchurWs(b1_t)
        schur_ct = SchurWs(c1_t)
        bal_ilo_t = Ref(1)
        bal_ihi_t = Ref(ma)
        bal_scale_t = ones(Float64, ma)
        new(ma, mb, b1, c1, a1, s2, t2, work1, work2, work3, work4, result, linsolve, schur_b, schur_c,
            bal_ilo, bal_ihi, bal_scale,
            b1_t, c1_t, s2_t, t2_t, schur_bt, schur_ct,
            bal_ilo_t, bal_ihi_t, bal_scale_t,
            nothing, nothing)
    end
end

# Swap rows i and j of matrix d in-place.
function _swap_rows!(d::AbstractMatrix, i::Int, j::Int)
    for k in axes(d, 2)
        d[i, k], d[j, k] = d[j, k], d[i, k]
    end
end

# Apply the row permutation + row scaling from gebal!('B', ...) to rows of d.
# Mirrors LAPACK's dgebak forward application.
function _balance_rows!(d::AbstractMatrix, ilo::Int, ihi::Int, scale::Vector{Float64})
    n = size(d, 1)
    # Permutation: from outside inward
    for i = n:-1:ihi+1
        j = Int(scale[i])
        j != i && _swap_rows!(d, i, j)
    end
    for i = 1:ilo-1
        j = Int(scale[i])
        j != i && _swap_rows!(d, i, j)
    end
    # Scaling: divide rows by scale factors
    for i = ilo:ihi
        s = scale[i]
        if s != 1.0
            for k in axes(d, 2)
                d[i, k] /= s
            end
        end
    end
    return d
end

# Undo the row permutation + row scaling (inverse of _balance_rows!).
function _unbalance_rows!(d::AbstractMatrix, ilo::Int, ihi::Int, scale::Vector{Float64})
    n = size(d, 1)
    # Undo scaling: multiply rows by scale factors
    for i = ilo:ihi
        s = scale[i]
        if s != 1.0
            for k in axes(d, 2)
                d[i, k] *= s
            end
        end
    end
    # Undo permutation: reverse order of _balance_rows!
    for i = ilo-1:-1:1
        j = Int(scale[i])
        j != i && _swap_rows!(d, i, j)
    end
    for i = ihi+1:n
        j = Int(scale[i])
        j != i && _swap_rows!(d, i, j)
    end
    return d
end

# After Schur decomposition, detect rank deficiency, reorder Schur form
# (zero eigenvalues to top), and build a dense LU factorization for the
# reduced r-block system. Returns RankDeficientData or nothing.
function _setup_rank_deficient(b1::Matrix{Float64}, vs::Matrix{Float64},
                               c1::Matrix{Float64}, order::Int)
    n = size(b1, 1)
    mc = size(c1, 1)
    tol = eps(Float64) * maximum(abs(b1[i,i]) for i in 1:n) * n
    n_deflated = count(i -> abs(b1[i,i]) < tol, 1:n)
    n_deflated == 0 && return nothing
    r = n - n_deflated
    # Reorder Schur form: zero eigenvalues to top
    select = Int64[abs(b1[i,i]) < tol ? 1 : 0 for i in 1:n]
    LAPACK.trsen!(select, b1, vs)
    # Save cross-block T12 before zeroing
    T12 = copy(b1[1:n_deflated, n_deflated+1:n])
    # Zero the entire top block (rows 1:n_deflated)
    b1[1:n_deflated, :] .= 0.0
    # Extract T22 (nonzero eigenvalue block)
    T22 = b1[n_deflated+1:n, n_deflated+1:n]
    # Compute kron(S, S) for order=2 (or S^⊗order)
    S_mat = Matrix(QuasiUpperTriangular(c1))
    kron_S = if order == 1
        S_mat
    else
        k = copy(S_mat)
        for _ in 2:order
            k = kron(k, S_mat)
        end
        k
    end
    mc_ord = mc^order
    # Build dense system: I + kron(kron(S',S',...), T22)
    kron_St = kron_S'  # = kron(S', S', ...)
    M_dense = kron(kron_St, T22)
    for i in 1:r*mc_ord
        M_dense[i, i] += 1.0
    end
    dense_lu = lu!(M_dense)
    return RankDeficientData(n_deflated, T12, kron_S, dense_lu)
end

function generalized_sylvester_factorize!(a::AbstractMatrix, b::AbstractMatrix, c::AbstractMatrix,
                                          order::Int64, ws::GeneralizedSylvesterWs;
                                          balance::Bool=true, deflate_tol::Float64=-1.0)
    copy!(ws.a1, a)
    copy!(ws.b1, b)
    copy!(ws.c1, c)
    factors = LinearAlgebra.LU(LAPACK.getrf!(ws.linsolve, ws.a1)...)
    ldiv!(factors, ws.b1)
    # Balance to improve Schur decomposition conditioning
    if balance
        ilo, ihi, scale = LAPACK.gebal!('B', ws.b1)
        ws.bal_ilo[] = ilo
        ws.bal_ihi[] = ihi
        copy!(ws.bal_scale, scale)
    else
        ws.bal_ilo[] = 1
        ws.bal_ihi[] = ws.ma
        fill!(ws.bal_scale, 1.0)
    end
    Schur(LAPACK.gees!(ws.schur_b, 'V', ws.b1)...)
    Schur(LAPACK.gees!(ws.schur_c, 'V', ws.c1)...)
    # Detect rank deficiency and set up dense fallback if needed
    if deflate_tol != 0.0
        ws.rddata = _setup_rank_deficient(ws.b1, ws.schur_b.vs, ws.c1, order)
    else
        ws.rddata = nothing
    end
    t = QuasiUpperTriangular(ws.b1)
    mul!(ws.t2, t, t)
    s = QuasiUpperTriangular(ws.c1)
    mul!(ws.s2, s, s)
end

# Dense solve for rank-deficient case in Schur coordinates.
# d is n × mc^order in Schur coordinates.
# Solves: Y + T * Y * kron(S,S) = F  where T has zero top-left block.
function _solve_rank_deficient!(d::AbstractMatrix, rd::RankDeficientData, mc_ord::Int)
    n = size(d, 1)
    nd = rd.n_deflated
    r = n - nd
    # Extract F2 (nonzero eigenvalue block RHS)
    F2 = d[nd+1:n, :]
    # Dense solve: (I + kron(kron(S',S'), T22)) * vec(Y2) = vec(F2)
    Y2_vec = rd.dense_lu \ vec(F2)
    Y2 = reshape(Y2_vec, r, mc_ord)
    # Correct Y1: Y1 = F1 - T12 * Y2 * kron(S, S)
    # d[1:nd, :] already contains F1
    Y2_kron = Y2 * rd.kron_S
    d[1:nd, :] .-= rd.T12 * Y2_kron
    # Write Y2 back
    d[nd+1:n, :] .= Y2
end

function generalized_sylvester_solve!(d::AbstractMatrix, order::Int64, ws::GeneralizedSylvesterWs)
    factors = LinearAlgebra.LU(ws.a1, ws.linsolve.ipiv, 0)
    ldiv!(factors, d)
    # Apply row balancing: d_bal = D^{-1}*P*d
    _balance_rows!(d, ws.bal_ilo[], ws.bal_ihi[], ws.bal_scale)
    t = QuasiUpperTriangular(ws.b1)
    s = QuasiUpperTriangular(ws.c1)
    at_mul_b_kron_c!(ws.result, ws.schur_b.vs, d, ws.schur_c.vs, order, ws.work2, ws.work3)
    copy!(d, ws.result)
    if ws.rddata !== nothing
        _solve_rank_deficient!(d, ws.rddata, size(ws.c1, 1)^order)
    else
        solve1!(1.0, order, t, ws.t2, s, ws.s2, vec(d), ws)
    end
    a_mul_b_kron_ct!(ws.result, ws.schur_b.vs, d, ws.schur_c.vs, order, ws.work2, ws.work3)
    copy!(d, reshape(ws.result, ws.ma, size(ws.c1, 1)^order))
    # Undo row balancing: X = P'*D*d
    _unbalance_rows!(d, ws.bal_ilo[], ws.bal_ihi[], ws.bal_scale)
end

# Solves the transposed system: A'·P + B'·P·(C'⊗...⊗C') = Ȳ
# Must be preceded by generalized_sylvester_factorize!, which caches LU(A) in ws.a1.
# factorize_transpose! computes new Schur decompositions of (A')⁻¹B' and C' into
# dedicated _t fields; the LU of A is reused at zero cost via transpose solve.
function generalized_sylvester_factorize_transpose!(b::AbstractMatrix, c::AbstractMatrix,
                                                    order::Int64, ws::GeneralizedSylvesterWs;
                                                    balance::Bool=true, deflate_tol::Float64=-1.0)
    factors = LinearAlgebra.LU(ws.a1, ws.linsolve.ipiv, 0)
    copy!(ws.b1_t, b')
    ldiv!(transpose(factors), ws.b1_t)
    copy!(ws.c1_t, c')
    if balance
        ilo, ihi, scale = LAPACK.gebal!('B', ws.b1_t)
        ws.bal_ilo_t[] = ilo
        ws.bal_ihi_t[] = ihi
        copy!(ws.bal_scale_t, scale)
    else
        ws.bal_ilo_t[] = 1
        ws.bal_ihi_t[] = ws.ma
        fill!(ws.bal_scale_t, 1.0)
    end
    Schur(LAPACK.gees!(ws.schur_bt, 'V', ws.b1_t)...)
    Schur(LAPACK.gees!(ws.schur_ct, 'V', ws.c1_t)...)
    if deflate_tol != 0.0
        ws.rddata_t = _setup_rank_deficient(ws.b1_t, ws.schur_bt.vs, ws.c1_t, order)
    else
        ws.rddata_t = nothing
    end
    t_t = QuasiUpperTriangular(ws.b1_t)
    mul!(ws.t2_t, t_t, t_t)
    s_t = QuasiUpperTriangular(ws.c1_t)
    mul!(ws.s2_t, s_t, s_t)
end

function generalized_sylvester_solve_transpose!(d::AbstractMatrix, order::Int64, ws::GeneralizedSylvesterWs)
    factors = LinearAlgebra.LU(ws.a1, ws.linsolve.ipiv, 0)
    ldiv!(transpose(factors), d)
    _balance_rows!(d, ws.bal_ilo_t[], ws.bal_ihi_t[], ws.bal_scale_t)
    t_t = QuasiUpperTriangular(ws.b1_t)
    s_t = QuasiUpperTriangular(ws.c1_t)
    at_mul_b_kron_c!(ws.result, ws.schur_bt.vs, d, ws.schur_ct.vs, order, ws.work2, ws.work3)
    copy!(d, ws.result)
    if ws.rddata_t !== nothing
        _solve_rank_deficient!(d, ws.rddata_t, size(ws.c1_t, 1)^order)
    else
        solve1!(1.0, order, t_t, ws.t2_t, s_t, ws.s2_t, vec(d), ws)
    end
    a_mul_b_kron_ct!(ws.result, ws.schur_bt.vs, d, ws.schur_ct.vs, order, ws.work2, ws.work3)
    copy!(d, reshape(ws.result, ws.ma, size(ws.c1_t, 1)^order))
    _unbalance_rows!(d, ws.bal_ilo_t[], ws.bal_ihi_t[], ws.bal_scale_t)
end

function generalized_sylvester_solver!(a::AbstractMatrix, b::AbstractMatrix, c::AbstractMatrix,
                                       d::AbstractMatrix, order::Int64, ws::GeneralizedSylvesterWs;
                                       balance::Bool=true, deflate_tol::Float64=-1.0)
    generalized_sylvester_factorize!(a, b, c, order, ws; balance, deflate_tol)
    generalized_sylvester_solve!(d, order, ws)
end


function solver!(t::QuasiUpperTriangular,s::QuasiUpperTriangular,d::AbstractVector,order::Int64,ws::GeneralizedSylvesterWs)
    s2 = QuasiUpperTriangular(s*s)
    t2 = QuasiUpperTriangular(t*t)
    solve1!(1.0,order,t,t2,s,s2,d,ws)
    d
end

"""
function solve1!(r, depth, t, t2, s, s2, d, ws)

solves (I + r*s^T ⊗ s^T ⊗ ... ⊗ s^T ⊗ t)x = d
where depth is the number of occurences of s^T
"""
function solve1!(r::Float64, depth::Int64, t::AbstractArray{Float64,2}, t2::AbstractArray{Float64,2}, s::AbstractArray{Float64,2}, s2::AbstractArray{Float64,2}, d::AbstractVector{Float64}, ws::GeneralizedSylvesterWs)
    m = size(t,2)
    n = size(s,1)
    if depth == 0
        I_plus_rA_ldiv_B!(r, t, d)
    else
        nd = m*n^(depth-1)
        nd2 = 2*nd
        drange1 = 1:nd
        drange2 = 1:nd2
        i = 1
        while i <= n
            if i == n || s[i+1,i] == 0
                dv = view(d,drange1)
                solve1!(r*s[i,i], depth-1, t, t2, s, s2, dv, ws)
                if i < n
                    solvi_real_eliminate!(i, n, nd, drange1, depth-1, r, t, s, d, ws)
                end
                drange1 = drange1 .+ nd
                drange2 = drange2 .+ nd
                i += 1
            else
                dv = view(d,drange2)
                solvii(r*s[i,i], r*s[i+1,i], r*s[i,i+1], depth-1, t, t2, s, s2, dv, ws)
                if i < n - 1
                    solvi_complex_eliminate!(i, n, nd, drange1, depth-1, r, t, s, d, ws)
                end
                drange1 = drange1 .+ nd2
                drange2 = drange2 .+ nd2
                i += 2
            end
        end
    end
end

"""
solvi_real_eliminate!(i::Int64, n::Int64, nd::Int64, drange::UnitRange{Int64},
                      depth::Int64, r::Float64, t::QuasiUpperTriangular, s::QuasiUpperTriangular,
                      d::AbstractVector, ws::GeneralizedSylvesterWs)
    updates d[k] with d[k] - ∑_{j=i+1}^n r*s[i, j]*(s ⊗ s ⊗ ... ⊗ s ⊗ t)*d with
    depth occurences of s
"""
function solvi_real_eliminate!(i::Int64, n::Int64, nd::Int64, drange::UnitRange{Int64},
                               depth::Int64, r::Float64, t::QuasiUpperTriangular, s::QuasiUpperTriangular,
                               d::AbstractVector, ws::GeneralizedSylvesterWs)
    work1 = ws.work1
    work2 = ws.work2
    work3 = ws.work3
    kron_at_kron_b_mul_c!(work1,1,s,depth,t,d,drange[1],work2,work3,1)
    k1 = drange[1] + nd
    @inbounds for j = i+1:n
        m = r*s[i,j]
        @simd for k2 = 1:nd
            d[k1] -= m*work1[k2]
            k1 += 1
        end
    end
end

function solvi_complex_eliminate!(i::Int64,n::Int64,nd::Int64,drange::UnitRange{Int64},
                                  depth::Int64,r::Float64,t::QuasiUpperTriangular,s::QuasiUpperTriangular,
                                  d::AbstractVector,ws::GeneralizedSylvesterWs)
    work1 = ws.work1
    work2 = ws.work2
    work3 = ws.work3
    kron_at_kron_b_mul_c!(work1, 1, s, depth, t, d, drange[1], work2, work3, 1)
    drange = drange .+ nd
    kron_at_kron_b_mul_c!(work1, nd+1, s, depth, t, d, drange[1], work2, work3, 1)
    k1 = drange[1] + nd
    @inbounds for j = i + 2 : n
        m1 = r*s[i,j]
        m2 = r*s[i+1,j]
        @simd for k2 = 1:nd
            d[k1] -= m1*work1[k2] + m2*work1[k2 + nd]
            k1 += 1
        end
    end
end

"""
function solveii!(alpha, beta1, beta2, depth, t, t2, s, s2, d, ws)

solves (I + G ⊗ s^T ⊗ s^T ⊗ ... ⊗ s^T ⊗ t)x = d
where depth is the number of occurences of s^T and
G = [alpha beta1; -beta2 alpha]
"""
function solvii(alpha::Float64,beta1::Float64,beta2::Float64,depth::Int64,
                t::QuasiUpperTriangular,t2::QuasiUpperTriangular,s::QuasiUpperTriangular,
                s2::QuasiUpperTriangular,d::AbstractVector,ws::GeneralizedSylvesterWs)
    m = size(t,2)
    n = size(s,1)
    nd = m*n^depth
    @assert !(beta1*beta2 > 0) "beta1*beta2 is positive"
    transformation1(alpha,beta1,beta2,depth,t,s,d,ws)
    dv = view(d,1:nd)
    solviip(alpha,sqrt(-beta1*beta2),depth,t,t2,s,s2,dv,ws)
    dv = view(d,nd+1:2*nd)
    solviip(alpha,sqrt(-beta1*beta2),depth,t,t2,s,s2,dv,ws)
end

function transformation1(a::Float64,b1::Float64,b2::Float64,depth::Int64,
                         t::QuasiUpperTriangular,s::QuasiUpperTriangular,
                         d::AbstractVector,ws::GeneralizedSylvesterWs)
    m = size(t, 2)
    n = size(s, 1)
    nd = m*n^depth
    copyto!(ws.work3, d)
    drange = 1:nd
    d1 = view(ws.work3, drange)
    d2 = view(ws.work3, drange .+ nd)
    work = view(ws.work2,drange)
    kron_at_kron_b_mul_c!(s,depth,t,d1,work)
    kron_at_kron_b_mul_c!(s,depth,t,d2,work)
    @inbounds @simd for i = drange
        d[i] += a*d1[i] - b1*d2[i]
        d[i+nd] += -b2*d1[i] + a*d2[i]
    end
end

diag_zero_sq = 1e-30

"""
function solveiip!(alpha, beta, depth, t, t2, s, s2, d, ws)

solves (I + 2*alpha*s^T ⊗ s^T ⊗ ... ⊗ s^T ⊗ t
          + (alpha^2 + beta^2)(s^2)^T ⊗ (s^2)^T ⊗ ... ⊗ (s^2)^T ⊗ t^2)x = d
where depth is the number of occurences of s^T
"""
function solviip(alpha::Float64,beta::Float64,depth::Int64,t::QuasiUpperTriangular,t2::QuasiUpperTriangular,
                 s::QuasiUpperTriangular,s2::QuasiUpperTriangular,d::AbstractVector,ws::GeneralizedSylvesterWs)
    m = size(t,2)
    n = size(s,1)
    if beta*beta < diag_zero_sq
        solve1!(alpha,depth,t,t2,s,s2,d,ws)
        solve1!(alpha,depth,t,t2,s,s2,d,ws)
        return
    end
    if depth == 0
        I_plus_rA_plus_sB_ldiv_C!(2*alpha,alpha*alpha+beta*beta,t,t2,d)
    else
        nd = m*n^(depth-1)
        nd2 = 2*nd
        drange1 = 1:nd
        drange2 = 1:nd2
        i = 1
        while i <= n
            if i == n || s[i+1,i] == 0
                dv = view(d,drange1)
                if s[i,i]*s[i,i]*(alpha*alpha+beta*beta) > diag_zero_sq
                    solviip(s[i,i]*alpha,s[i,i]*beta,depth-1,t,t2,s,s2,dv,ws)
                end
                if i < n
                    solviip_real_eliminate!(i,n,nd,drange1,depth - 1,alpha,beta,t,t2,s,s2,d,ws)
                end
                drange1 = drange1 .+ nd
                drange2 = drange2 .+ nd
                i += 1
            else
                dv = view(d,drange2)
                solviip2(alpha,beta,s[i,i],s[i+1,i],s[i,i+1],depth - 1,t,t2,s,s2,dv,ws)
                if i < n - 1
                    solviip_complex_eliminate!(i,n,nd,drange1,depth-1,alpha,beta,t,t2,s,s2,d,ws)
                end
                drange1 = drange1 .+ nd2
                drange2 = drange2 .+ nd2
                i += 2
            end
        end
    end
end

function solviip_real_eliminate!(i::Int64,n::Int64,nd::Int64,drange::UnitRange{Int64},
                                 depth::Int64,alpha::Float64,beta::Float64,t::QuasiUpperTriangular,
                                 t2::QuasiUpperTriangular,s::QuasiUpperTriangular,
                                 s2::QuasiUpperTriangular,d::AbstractVector,ws::GeneralizedSylvesterWs)
    y1 = view(ws.work1,drange)
    y2 = view(ws.work1,drange .+ nd)
    copyto!(y1,1,d,drange[1],nd)
    copyto!(y2,1,d,drange[1],nd)
    work = view(ws.work2,1:length(drange))
    kron_at_kron_b_mul_c!(s,depth,t,y1,work)
    kron_at_kron_b_mul_c!(s2,depth,t2,y2,work)
    k1 = drange[1] + nd
    @inbounds for j = i+1:n
        m1 = 2*alpha*s[i,j]
        m2 = (alpha*alpha+beta*beta)*s2[i,j]
        @simd for k2 = 1:nd
            d[k1] -= m1*y1[k2] + m2*y2[k2]
            k1 += 1
        end
    end
end

"""
function solveiip2!(alpha, beta, gamma, delta1, delta2, depth, t, t2, s, s2, d, ws)

    solves (I + 2*alpha*G ⊗ s^T ⊗ s^T ⊗ ... ⊗ s^T ⊗ t
          + (alpha^2 + beta^2)*G^2 ⊗ (s^2)^T ⊗ (s^2)^T ⊗ ... ⊗ (s^2)^T ⊗ t^2)x = d
where depth is the number of occurences of s^T and
G = [gamma delta1; -delta2 gammaa]
"""
function solviip2(alpha::Float64,beta::Float64,gamma::Float64,delta1::Float64,delta2::Float64,
                  depth::Int64,t::QuasiUpperTriangular,t2::QuasiUpperTriangular,
                  s::QuasiUpperTriangular,s2::QuasiUpperTriangular,d::AbstractVector,ws::GeneralizedSylvesterWs)
    m = size(t,2)
    n = size(s,1)
    aspds = alpha*alpha + beta*beta
    gspds = gamma*gamma - delta1*delta2
    nd = m*n^depth
    dv1 = view(d,1:nd)
    dv2 = view(d,nd+1:2*nd)
    transform2(alpha, beta, gamma, -delta1, -delta2, nd, depth, t, t2, s, s2, d,ws)

    delta = sqrt(-delta1*delta2)
    a1 = alpha*gamma - beta*delta
    b1 = alpha*delta + gamma*beta
    a2 = alpha*gamma + beta*delta
    b2 = alpha*delta - gamma*beta
    solviip(a2, b2, depth, t, t2, s, s2, dv1, ws);
    solviip(a1, b1, depth, t, t2, s, s2, dv1, ws);
    solviip(a2, b2, depth, t, t2, s, s2, dv2, ws);
    solviip(a1, b1, depth, t, t2, s, s2, dv2, ws);
end

"""
function transform2(alpha, beta, gamma, delta1, delta2, nd, depth, t, t2, s, s2, d, ws)

mutates [d1, d2] by computing
    [d1, d2] = (I + 2*alpha*[gamma delta1;delta2 gamma] ⊗ ( s^T⊗ s^T ⊗ ... ⊗ s^T ⊗ t)
                + (alpha^2 + beta^2)*[gamma delta1;delta2 gamma]^2 ⊗( s^T⊗ s^T ⊗ ... ⊗ s^T ⊗ t))*[d1; d2]
"""
function transform2(alpha::Float64, beta::Float64, gamma::Float64, delta1::Float64, delta2::Float64,
                    nd::Int64, depth::Int64, t::QuasiUpperTriangular, t2::QuasiUpperTriangular,
                    s::QuasiUpperTriangular, s2::QuasiUpperTriangular, d::AbstractVector{Float64}, ws::GeneralizedSylvesterWs)
    d1 = ws.work1
    kron_at_kron_b_mul_c!(d1,1,s,depth,t,d,1,ws.work3,ws.work4,1)
    kron_at_kron_b_mul_c!(d1,nd+1,s,depth,t,d,nd+1,ws.work3,ws.work4,1)

    d2 = ws.work2
    kron_at_kron_b_mul_c!(d2,1,s2,depth,t2,d,1,ws.work3,ws.work4,1)
    kron_at_kron_b_mul_c!(d2,nd+1,s2,depth,t2,d,nd+1,ws.work3,ws.work4,1)

    m1 = 2*alpha*gamma
    m2 = 2*alpha*delta1
    m3 = 2*alpha*delta2
    aspds = alpha*alpha + beta*beta;
    gspds = gamma*gamma + delta1*delta2;
    m11 = aspds*gspds
    m22 = 2*aspds*gamma*delta1
    m33 = 2*aspds*gamma*delta2
    @inbounds @simd for i = 1:nd
        d1i = d1[i]
        d2i = d2[i]
        d1ind = d1[i+nd]
        d2ind = d2[i+nd]
        d[i] +=  m1*d1i + m2*d1ind +  m11*d2i + m22*d2ind
        d[i+nd] +=  m3*d1i + m1*d1ind + m33*d2i + m11*d2ind
    end
end

"""
    solviip_complex_eliminate!(i,n,nd,drange,depth,alpha,beta,t,t2,s,s2,d)

perfoms elimination after solving for a complex diagonal block of size 2*n^depth

d n^(depth+2) x 1

The solution is stored in d[drange; drange + nd]

The function updates d[i*nd+1:n*nd]
"""
function solviip_complex_eliminate!(i::Int64,n::Int64,nd::Int64,drange::UnitRange{Int64},depth::Int64,
                                    alpha::Float64,beta::Float64,t::QuasiUpperTriangular,t2::QuasiUpperTriangular,
                                    s::QuasiUpperTriangular,s2::QuasiUpperTriangular,d::AbstractVector,ws::GeneralizedSylvesterWs)
    y11 = view(ws.work1,drange)
    y12 = view(ws.work1, drange .+ nd)
    copyto!(y11,1,d,drange[1],nd)
    copyto!(y12,1,d,drange[1],nd)
    drange = drange .+ nd
    y21 = view(ws.work2,drange)
    y22 = view(ws.work2, drange .+ nd)
    copyto!(y21,1,d,drange[1],nd)
    copyto!(y22,1,d,drange[1],nd)
    work = view(ws.work3,drange)

    kron_at_kron_b_mul_c!(s,depth,t,y11,work)
    kron_at_kron_b_mul_c!(s2,depth,t2,y12,work)
    kron_at_kron_b_mul_c!(s,depth,t,y21,work)
    kron_at_kron_b_mul_c!(s2,depth,t2,y22,work)

    alpha2beta2 = alpha*alpha + beta*beta
    k1 = drange[1] + nd
    @inbounds for j = i+2:n
        m1 = 2*alpha*s[i,j]
        m2 = alpha2beta2*s2[i,j]
        m3 = 2*alpha*s[i+1,j]
        m4 = alpha2beta2*s2[i+1,j]
        @simd for k2 = 1:nd
            d[k1] -= m1*y11[k2] + m2*y12[k2] + m3*y21[k2] + m4*y22[k2]
            k1 += 1
        end
    end
end

end
