import GeneralizedSylvesterSolver: GeneralizedSylvesterWs, solvi_real_eliminate!,
    QuasiUpperTriangular, solve1!, transformation1, transform2,
    generalized_sylvester_solver!, generalized_sylvester_factorize!, generalized_sylvester_solve!,
    generalized_sylvester_factorize_transpose!, generalized_sylvester_solve_transpose!,
    solvi_complex_eliminate!, solviip_complex_eliminate!, solver!,
    solvii, solviip, solviip2, solviip_real_eliminate!

using LinearAlgebra    
using Random
using Test

    
function kron_power(x,order)
    if order == 0
        return 1
    elseif order == 1
        return x
    else
        m, n = size(x)
        y = Matrix{Float64}(undef, m^order, n^order)
        y1 = similar(y)
        v = view(y, 1:m, 1:n)
        v1 = view(y1, 1:m, 1:n)
        v .= x
        for i = 1:(order-1)
            tmp = v1
            v1 = v
            v = tmp
            v = view(v.parent, 1:m^(i + 1), 1:n^(i + 1))
            kron!(v, v1, x)
        end
    end
    return v.parent
end

sreal = [2.0 1.0; 0.0 -3.0]
scplx = [-2.0 3.0; -1.0 -2.0]
@assert scplx[1,1] == scplx[2,2] "not a Schur block"
@assert !(scplx[2,1]*scplx[1,2] > 0) "not a Schur block"

treal = [1.0 3.0; 0.0 -2.0]
tcplx = [1.0 3.0; 1.0 1.0]
o2 = zeros(2,2)

matrices= [ (sreal, treal),
            (scplx, treal),
            (scplx, tcplx),
            (scplx, tcplx),
            ([sreal scplx; o2 sreal], [treal treal; o2 treal]),
            ([sreal scplx; o2 scplx], [treal treal; o2 treal]),
            ([scplx scplx; o2 sreal], [treal treal; o2 treal]),
            ([sreal scplx; o2 sreal], [treal tcplx; o2 treal]),
            ([sreal scplx; o2 scplx], [treal tcplx; o2 treal]),
            ([scplx scplx; o2 sreal], [treal tcplx; o2 treal]),
            ([sreal scplx; o2 sreal], [tcplx tcplx; o2 tcplx]),
            ([sreal scplx; o2 scplx], [tcplx tcplx; o2 tcplx]),
            ([scplx scplx; o2 sreal], [tcplx tcplx; o2 tcplx])
           ]


@testset "Generalized Sylvester Equation $depth, $mat " for depth = 0:4, mat in matrices
    s = QuasiUpperTriangular(mat[1])
    t = QuasiUpperTriangular(mat[2])
    s2 = QuasiUpperTriangular(s*s)
    t2 = QuasiUpperTriangular(t*t)
    n = size(s,1)
    m = size(t,1)
    d_orig = randn(m*n^depth)
    r = 1.5
        
    ws = GeneralizedSylvesterWs(m, m, n, depth)

    @testset "kron_power" begin
        @test kron_power(s,1) == s
        @test kron_power(s,3) ≈ kron(kron(s,s),s) 
    end

    if depth > 0
        nd = m*n^(depth-1)
        @testset "SOLVI_REAL_ELIMINATE" begin
            drange = 1:nd
            d = copy(d_orig)
            for index = 1:(n-1)
                d_target = d[(index*nd + 1): m*n^depth]  - r*kron(s[index,(index+1):n],kron(kron_power(s',depth-1),t))*d[(index-1)*nd .+ (1:m*n^(depth-1))]
                solvi_real_eliminate!(index, n, nd, drange, depth-1, r, t, s, d, ws)
                @test d_target ≈ d[(index*nd + 1): m*n^depth]
                drange = drange .+ nd
            end
        end

        @testset "SOLVI_COMPLEX_ELIMINATE" begin
            drange = 1:nd
            d = copy(d_orig)
            for index = 1:2:(n-2)
                d_target = (d[(index+1)*nd .+ 1: m*n^depth]  - r*kron(s[index,(index+2):n],kron(kron_power(s',depth-1),t))*d[(index-1)*nd .+ (1:m*n^(depth-1))]
                            - r*kron(s[index + 1,(index+2):n],kron(kron_power(s',depth-1),t))*d[(index-1)*nd + m*n^(depth-1) .+ (1:m*n^(depth-1))])
                solvi_complex_eliminate!(index, n, nd, drange, depth-1, r, t, s, d, ws)
                @test d_target ≈ d[(index+1)*nd .+ 1: m*n^depth]
                drange = drange .+ 2*nd
            end
        end

        @testset "SOLVIIP_REAL_ELIMINATE" begin 
            drange = 1:nd
            d = copy(d_orig)
            alpha = randn()
            beta = randn()
            for index = 1:(n-1)
                d_target = (d[(index*nd + 1): m*n^depth]
                            - 2*alpha*kron(s[index,(index+1):n],kron(kron_power(s',depth-1),t))*d[(index-1)*nd .+ (1:m*n^(depth-1))]
                            - (alpha^2 + beta^2)*kron(s2[index,(index+1):n],kron(kron_power(s2',depth-1),t2))*d[(index-1)*nd .+ (1:m*n^(depth-1))])
                solviip_real_eliminate!(index, n, nd, drange, depth-1, alpha, beta, t, t2, s, s2, d, ws)
                @test d_target ≈ d[(index*nd + 1): m*n^depth]
                drange = drange .+ nd
            end
        end

        @testset "SOLVIIP_COMPLEX_ELIMINATE" begin
            drange = 1:nd
            d = copy(d_orig)
            alpha = randn()
            beta = randn()
            for index = 1:2:(n-2)
                d_target = (d[(index+1)*nd + 1: m*n^depth]
                            - 2*alpha*kron(s[index,(index+2):n],kron(kron_power(s',depth-1),t))*d[(index-1)*nd .+ (1:m*n^(depth-1))]
                            - (alpha^2 + beta^2)*kron(s2[index,(index+2):n],kron(kron_power(s2',depth-1),t2))*d[(index-1)*nd .+ (1:m*n^(depth-1))]
                            - 2*alpha*kron(s[index + 1,(index+2):n],kron(kron_power(s',depth-1),t))*d[(index-1)*nd + m*n^(depth-1) .+ (1:m*n^(depth-1))]
                            - (alpha^2 + beta^2)*kron(s2[index + 1,(index+2):n],kron(kron_power(s2',depth-1),t2))*d[(index-1)*nd  + m*n^(depth-1) .+ (1:m*n^(depth-1))])
                solviip_complex_eliminate!(index, n, nd, drange, depth-1, alpha, beta, t, t2, s, s2, d, ws)
                @test d_target ≈ d[(index+1)*nd .+ 1: m*n^depth]
                drange = drange .+ 2*nd
            end
        end

        @testset "SOLVEIIP2" begin
            alpha = randn()
            beta = randn()
            a = randn()
            b1 = -rand()
            b2 = rand()
            G = [a b1; b2 a]
            nd1 = 2*m*n^(depth-1)
            d = copy(d_orig[1:nd1])
            d_target = (I(nd1) + 2*alpha*kron(kron(G',kron_power(s',depth-1)),t)
                        + (alpha*alpha + beta*beta)*kron(kron(G'*G',kron_power(s2',depth-1)),t2))\d
            solviip2(alpha, beta, a, b2, b1, depth - 1, t, t2, s, s2, d, ws)
            @test d ≈ d_target
        end
        
        @testset "TRANSFORMATION1" begin
            a = randn()
            b1 = -rand()
            b2 = rand()
            nd1 = 2*m*n^(depth -1)
            d = copy(d_orig[1:nd1])
            d_target = d + kron([a -b1; -b2 a], kron(kron_power(s', depth - 1), t))*d
            transformation1(a, b1, b2, depth - 1, t, s, d, ws)
            @test d ≈ d_target
        end

        @testset "TRANSFORMATION2" begin
            a = randn()
            b1 = -rand()
            b2 = rand()
            r1 = randn()
            r2 = randn()
            nd1 = 2*m*n^(depth -1)
            d = copy(d_orig[1:nd1])
            d_target = (I(nd1) + 2*r1*kron([a b1; b2 a],kron(kron_power(s', depth -1), t))
                        + (r1*r1 + r2*r2)*kron([a b1; b2 a]*[a b1; b2 a], kron(kron_power(s2', depth - 1), t2)))*d
            transform2(r1, r2, a, b1, b2, m*n^(depth - 1), depth - 1, t, t2, s, s2, d, ws)
            @test d ≈ d_target
        end

        @testset "SOLVEII" begin
            alpha = randn()
            beta1 = -rand()
            beta2 = rand()
            G = [alpha beta1; beta2 alpha]
            nd1 = 2*m*n^(depth -1)
            d = copy(d_orig[1:nd1])
            d_target = (I(nd1) + kron(kron(G, kron_power(s',depth-1)), t))\d
            solvii(alpha, beta1, beta2, depth - 1, t, t2, s, s2, d, ws)
            @test d ≈ d_target
            beta1 = rand()
            beta2 = -rand()
            G = [alpha beta1; beta2 alpha]
            d = copy(d_orig[1:nd1])
            d_target = (I(nd1) + kron(kron(G, kron_power(s',depth-1)), t))\d
            solvii(alpha, beta1, beta2, depth - 1, t, t2, s, s2, d, ws)
            @test d ≈ d_target
        end
        
        @testset "SOLVEIIP beta == 0.0" begin
            alpha = randn()
            beta1 = 0.0
            nd = m*n^(depth - 1)
            d = copy(d_orig[1:nd])
            d_target = (I(nd) + 2*alpha*kron(kron_power(s',depth - 1),t) + (alpha*alpha + beta1*beta1)*kron(kron_power(s2', depth - 1),t2))\d
            solviip(alpha, beta1, depth - 1, t, t2, s, s2, d, ws)
            @test d ≈ d_target
        end        

        @testset "SOLVEIIP beta == 2.0" begin
            alpha = randn()
            beta1 = randn()
            nd = m*n^(depth - 1)
            d = copy(d_orig[1:nd])
            d_target = (I(nd) + 2*alpha*kron(kron_power(s',depth - 1),t) + (alpha*alpha + beta1*beta1)*kron(kron_power(s2', depth - 1),t2))\d
            solviip(alpha, beta1, depth - 1, t, t2, s, s2, d, ws)
            @test d ≈ d_target
        end        
    end
    
    @testset "SOLVE1" begin
        d = copy(d_orig)
        r = 1.0
        d_target = (I(m*n^depth) + r*kron(kron_power(s', depth), t))\d
        solve1!(r, depth, t, t2, s, s2, d, ws)
        @test d ≈ d_target
    end

end


n = 4
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
t = QuasiUpperTriangular(schur(a\b).T)
s = QuasiUpperTriangular(schur(c).T)
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
depth = 1
d_orig = randn(n^(depth+1))
d = copy(d_orig)
ws = GeneralizedSylvesterWs(n,n,n,depth)
solver!(t,s,d,depth,ws)
d_target = (I(n^(depth+1)) + kron(s',t))\d_orig
@test d ≈ d_target

n = 4
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
t = QuasiUpperTriangular(schur(a\b).T)
s = QuasiUpperTriangular(schur(c).T)
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
depth = 3
d_orig = randn(n^(depth+1))
d = copy(d_orig)
ws = GeneralizedSylvesterWs(n,n,n,depth)
solver!(t,s,d,depth,ws)
d_target = (I(n^(depth+1)) + kron(s',kron(kron(s',s'),t)))\d_orig
@test d ≈ d_target

n = 4
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
t = QuasiUpperTriangular(schur(a\b).T)
s = QuasiUpperTriangular(schur(c).T)
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
depth = 3
ws = GeneralizedSylvesterWs(n,n,n,depth)
d_orig = randn(n^(depth+1))
d = copy(d_orig)
solver!(t,s,d,depth,ws)
d_target = (I(n^(depth+1)) + kron(s',kron(kron(s',s'),t)))\d_orig
@test d ≈ d_target


n1 = 4
n2 = 3
a_orig = randn(n1,n1)
b_orig = randn(n1,n1)
c_orig = randn(n2,n2)

depth = 1
ws = GeneralizedSylvesterWs(n1,n1,n2,depth)
d_orig = randn(n1,n2^depth)
a = copy(a_orig)
b = copy(b_orig)
c = copy(c_orig)
d = copy(d_orig)

d = reshape(d, 4, 3)
generalized_sylvester_solver!(a,b,c,d,1,ws)
@test a_orig*d + b_orig*d*c_orig ≈ d_orig
@test d ≈ reshape((kron(I(n2^depth),a_orig) + kron(c_orig',b_orig))\vec(d_orig),n1,n2^depth)

depth = 2
ws = GeneralizedSylvesterWs(n1,n1,n2,depth)
d_orig = randn(n1,n2^depth)
a = copy(a_orig)
b = copy(b_orig)
c = copy(c_orig)
d = copy(d_orig)

d = reshape(d, 4, 9)
generalized_sylvester_solver!(a,b,c,d,2,ws)
@test a_orig*d + b_orig*d*kron(c_orig,c_orig) ≈ d_orig
@test d ≈ reshape((kron(I(n2^depth),a_orig) + kron(kron(c_orig',c_orig'),b_orig))\vec(d_orig),n1,n2^depth)

n1 = 4
n2 = 3
a_orig2 = randn(n1,n1)
b_orig2 = randn(n1,n1)
c_orig2 = randn(n2,n2)
depth2 = 2
d_orig2 = randn(n1,n2^depth2)
ws1 = GeneralizedSylvesterWs(n1,n1,n2,depth2)
ws2 = GeneralizedSylvesterWs(n1,n1,n2,depth2)
d1 = copy(d_orig2)
d2 = copy(d_orig2)
generalized_sylvester_solver!(copy(a_orig2),copy(b_orig2),copy(c_orig2),d1,depth2,ws1)
generalized_sylvester_factorize!(copy(a_orig2),copy(b_orig2),copy(c_orig2),depth2,ws2)
generalized_sylvester_solve!(d2,depth2,ws2)
@test d1 ≈ d2

# Test factorize_transpose! + solve_transpose!
# Solves: A'·P + B'·P·(C'⊗C') = Ȳ
# Vectorized: (I⊗A' + (C⊗C)⊗B')·vec(P) = vec(Ȳ)
for depth_t in (1, 2)
    local n1_t = 4; local n2_t = 3
    local a_t = randn(n1_t,n1_t); local b_t = randn(n1_t,n1_t); local c_t = randn(n2_t,n2_t)
    local ybar_orig = randn(n1_t, n2_t^depth_t)
    local ybar = copy(ybar_orig)
    local ws_t = GeneralizedSylvesterWs(n1_t,n1_t,n2_t,depth_t)
    generalized_sylvester_factorize!(copy(a_t),copy(b_t),copy(c_t),depth_t,ws_t)
    generalized_sylvester_factorize_transpose!(b_t,c_t,depth_t,ws_t)
    generalized_sylvester_solve_transpose!(ybar,depth_t,ws_t)
    kron_c = depth_t == 1 ? c_t : kron(c_t,c_t)
    P_expected = reshape((kron(I(n2_t^depth_t),a_t') + kron(kron_c,b_t')) \ vec(ybar_orig), n1_t, n2_t^depth_t)
    @test ybar ≈ P_expected
end

@testset "rank-deficient b matrix (dense fallback)" begin
    # Construct a problem where b is rank-deficient (rank r << n).
    # The dense fallback (Schur reorder + direct solve of r-block) handles this.
    Random.seed!(2)
    n1 = 6; n2 = 3; r = 2

    a_orig = Matrix{Float64}(I, n1, n1)
    U = 0.3 * randn(n1, r)
    V = 0.3 * randn(n1, r)
    b_orig = U * V'
    c_orig = 0.3 * randn(n2, n2)

    for depth in 1:2
        ws = GeneralizedSylvesterWs(n1, n1, n2, depth)
        d_orig = randn(n1, n2^depth)
        C_kron = depth == 1 ? c_orig : kron(c_orig, c_orig)

        # With dense fallback opted in (deflate_tol=-1.0), the solve succeeds
        a, b, c, d = copy(a_orig), copy(b_orig), copy(c_orig), copy(d_orig)
        generalized_sylvester_solver!(a, b, c, d, depth, ws; balance=true, deflate_tol=-1.0)
        @test a_orig * d + b_orig * d * C_kron ≈ d_orig

        # With upstream-matching defaults (no fallback), the recursive solver fails
        a3, b3, c3, d3 = copy(a_orig), copy(b_orig), copy(c_orig), copy(d_orig)
        generalized_sylvester_solver!(a3, b3, c3, d3, depth, ws)
        res_raw = norm(a_orig * d3 + b_orig * d3 * C_kron - d_orig) / norm(d_orig)
        @test res_raw > 0.1
    end
end

@testset "DSGE model fixtures (order=2)" begin
    include(joinpath(@__DIR__, "dsge_fixtures.jl"))
    # RBC and SGU have valid E_SYL; test them directly
    for (label, A, B, C, E) in [
        ("RBC",  RBC_A_SYL,  RBC_C_SYL,  RBC_H_X,  RBC_E_SYL),
        ("SGU",  SGU_A_SYL,  SGU_C_SYL,  SGU_H_X,  SGU_E_SYL),
    ]
        @testset "$label" begin
            n, nx = size(A, 1), size(C, 1)
            ws = GeneralizedSylvesterWs(n, n, nx, 2)
            a, b, c, d = copy(A), copy(B), copy(C), copy(E)
            generalized_sylvester_solver!(a, b, c, d, 2, ws)
            C_kron = kron(C, C)
            residual = norm(A * d + B * d * C_kron - E) / norm(E)
            @test residual < 1e-8
        end
    end
    # FVGQ: rank-deficient B (rank 6 out of 38). Test with synthetic RHS
    # since the fixture E_SYL is zero (extracted from a failed solve).
    @testset "FVGQ (synthetic RHS)" begin
        A, B, C = FVGQ_A_SYL, FVGQ_C_SYL, FVGQ_H_X
        n, nx = size(A, 1), size(C, 1)
        Random.seed!(1234)
        X_true = randn(n, nx^2) * 0.01
        C_kron = kron(C, C)
        E = A * X_true + B * X_true * C_kron
        ws = GeneralizedSylvesterWs(n, n, nx, 2)
        a, b, c, d = copy(A), copy(B), copy(C), copy(E)
        generalized_sylvester_solver!(a, b, c, d, 2, ws; balance=true, deflate_tol=-1.0)
        residual = norm(A * d + B * d * C_kron - E) / norm(E)
        @test residual < 1e-7
    end
end

function f(t,s,d,depth,ws)
    for i = 1:100
        solver!(t,s,d,depth,ws)
    end
end

#@profile  f(t,s,d,depth,ws)
#Profile.print(combine=true,sortedby=:count)

