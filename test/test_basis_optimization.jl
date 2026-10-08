using HTTN
using TensorKit
using LinearAlgebra
using Random
using Test

@testset "Basis optimization regressions" begin
    Random.seed!(82413)
    model(n = 3, k = 1; ξ = zeros(ComplexF64, k + 1), extras = (;)) = MassiveSchwingerModel(
        merge(
            (
                kMax = k, nMax = n, nMaxZM = n, truncMethod = 5,
                modeOrdering = true, bogoliubovRot = true, bogParameters = ξ,
            ), extras
        ),
        (θ = Float64(pi), m = 0.1, M = 1 / sqrt(pi), L = 100.0)
    )
    matrix(H) = reshape(mpo2mat(H), prod(dim(space(t, 2)) for t in H), :)
    pair(A, B) = HTTN.convertLocalOperatorsToTwoBodyGate([A, B])

    @testset "Squeezing and entropy" begin
        m = model(6)
        PL, PR = m.physSpaces[2:3]
        Kp = pair(HTTN.localCreationOp(-1, PL), HTTN.localCreationOp(1, PR))
        Km = Kp'
        NN = pair(HTTN.localNumberOp(PL), HTTN.localIdentityOp(PR)) +
            pair(HTTN.localIdentityOp(PL), HTTN.localNumberOp(PR)) + one(Kp)
        for ξ in (0.0, 0.6, 1.0, 0.2 + 0.1im)
            μ, ν = HTTN.convertSqueezingParameter(ξ)
            # The nilpotent outer series terminate and provide an independent reference.
            reference = HTTN.matrixExponentialSeries(-ν / μ * Kp, 6) *
                exp(-log(μ) * NN) * HTTN.matrixExponentialSeries(conj(ν) / μ * Km, 6)
            @test squeezingOp(ξ, 6, -1, 1, PL, PR) ≈ reference atol = 1.0e-12
            P0 = m.physSpaces[1]
            An = TensorMap(HTTN.getAnnihilationOperator(6), P0, P0)
            N = An' * An
            reference0 = HTTN.matrixExponentialSeries(-ν / (2μ) * An'^2, 3) *
                exp(-log(μ) * (N + 0.5one(N))) *
                HTTN.matrixExponentialSeries(conj(ν) / (2μ) * An^2, 3)
            @test singleSqueezingOp(ξ, 6, P0) ≈ reference0 atol = 1.0e-12
        end
        δ = 1.0e-6
        derivative = (
            squeezingOp(0.6 + δ, 6, -1, 1, PL, PR) -
                squeezingOp(0.6 - δ, 6, -1, 1, PL, PR)
        ) / (2δ)
        @test HTTN.gradient_squeezing_operator(0.6, 6, -1, 1, PL, PR) ≈ derivative rtol = 1.0e-8
        p = initializeVacuumMPS(m)
        AC2 = permute(p[2] * permute(p[3], ((1,), (2, 3))), ((1, 2), (3, 4)))
        @test HTTN.computeRenyiEntropy(AC2) ≈ HTTN.computeRenyiEntropy(0.5AC2) atol = 1.0e-14
        @test HTTN.computeEntropy(AC2) ≈ 0 atol = 1.0e-14
        AC2 = HTTN.applyTwoModeTransformation(squeezingOp(0.15, 6, -1, 1, PL, PR), AC2)
        @test HTTN.computeEntropy(AC2) ≈ HTTN.computeEntropy(0.5AC2) atol = 1.0e-14
        cost(ξ) = HTTN.findDisentanglingRotation(ξ, 6, -1, 1, PL, PR, AC2)
        for ξ in (0.0, 0.25, 0.2 + 0.1im)
            _, gradient = HTTN.value_and_gradient(ξ, 6, -1, 1, PL, PR, AC2)
            @test real(gradient) ≈ (cost(ξ + δ) - cost(ξ - δ)) / (2δ) atol = 1.0e-7
            if ξ isa Complex
                @test imag(gradient) ≈ (cost(ξ + δ * im) - cost(ξ - δ * im)) / (2δ) atol = 1.0e-7
            else
                @test HTTN.analyticGradientCostFunction(ξ, 6, -1, 1, PL, PR, AC2) ≈ gradient atol = 1.0e-7
            end
        end
    end

    @testset "Complex squeezing composition" begin
        m = model(12)
        PL, PR = m.physSpaces[2:3]
        for (ξ, δξ) in ((0.15, 0.12im), (0.1im, 0.12 - 0.03im), (0.1, -0.1), (0im, 0.2im))
            U, newXi = HTTN.squeezingUpdate(ξ, δξ, 12, -1, 1, PL, PR)
            composed = U * squeezingOp(ξ, 12, -1, 1, PL, PR)
            reference = squeezingOp(newXi, 12, -1, 1, PL, PR)
            p = initializeVacuumMPS(m)
            AC2 = permute(p[2] * permute(p[3], ((1,), (2, 3))), ((1, 2), (3, 4)))
            # Compare on a low-occupation state, away from the projection boundary.
            @test norm(applyTwoModeTransformation(composed - reference, AC2)) < 1.0e-10
        end
    end

    @testset "Model and MPO updates" begin
        m = model(; extras = (decouplePairs = true,))
        ξ = ComplexF64[0.1im, 0.2 + 0.1im]
        updated = updateBogoliubovParameters(m; bogoliubovRot = true, bogParameters = ξ)
        @test updated.modelParameters.truncationParameters.decouplePairs
        ξ[1] = 0
        @test updated.modelParameters.truncationParameters.bogParameters[1] == 0.1im
        for ξ in ([0.2, 0.0], [0.0, 0.2], [0.1im, 0.2 + 0.1im])
            m = model(; ξ)
            @test matrix(generate_H1(m; localOp = "vertexOp")) ≈ matrix(generate_H1(m; localOp = "displacementOp"))
        end
        m = model(3, 0)
        @test matrix(generate_H0(m)) ≈ matrix(generate_H0(updateBogoliubovParameters(m; bogoliubovRot = false, bogParameters = [0.0])))
        m = model()
        H = HTTN.constructIdentityMPO(m.physSpaces, U1Space(0 => 1))
        S = squeezingOp(0.3, 3, -1, 1, m.physSpaces[2], m.physSpaces[3])
        W = kron(reshape(convert(Array, S), 16, 16), Matrix{ComplexF64}(I, 4, 4))
        transformed, _ = bogTransformMPO(H, [0.0, 0.3]; truncErr = 1.0e-14)
        @test matrix(transformed) ≈ W * matrix(H) * W' atol = 1.0e-11
        @test HTTN.maxLinkDimsMPO(transformed) > HTTN.maxLinkDimsMPO(H)
        @test_throws ArgumentError bogTransformMPO(H, [0.1, 0.0])
        E = HTTN.SparseLocalOp([HTTN.localIdentityOp(m.physSpaces[1])])
        @test length(E) == 1
        @test size(E) == (1,)
        @test first(E) == E[1]
        @test HTTN.getPhysicalSpace(E, 1) == m.physSpaces[1]
        @test HTTN.getKroneckerDeltaSpace(E, 1) == space(E[1], 3)'
        p = initializeVacuumMPS(m)
        @test typeof(similar(p)) == typeof(p)
        @test size(similar(p)) == size(p)
        @test typeof(similar(H)) == typeof(H)
        @test size(similar(H)) == size(H)
    end

    @testset "TDVP time convention and gauge" begin
        m = model(2, 2)
        H = generate_MPO_mS(m)
        vs = constructVirtSpaces(m.physSpaces, U1Space(0 => 1), U1Space(0 => 1); removeDegeneracy = true)
        p = SparseMPS(randn, ComplexF64, m.physSpaces, vs; normalizeMPS = true)
        for algorithm in (TDVP1, TDVP2, TDVP2BO)
            alg = algorithm(; extendBasis = false, bondDim = 128, truncErrT = 1.0e-12)
            a, _, _, _ = perform_timestep!(copy(p), H, 0.05, alg)
            b = copy(p)
            HTTN.orthogonalizeMPS!(b, 4)
            b, _, _, _ = perform_timestep!(b, H, 0.05, alg)
            c, _, _, _ = perform_timestep!(copy(p), H, 0.05 + 0im, alg)
            @test vec(mps2vec(a)) ≈ vec(mps2vec(b)) atol = 1.0e-10
            @test vec(mps2vec(a)) ≈ vec(mps2vec(c)) atol = 1.0e-10
        end
        m = model(3, 0)
        H = generate_MPO_mS(m)
        p = SparseMPS(randn, ComplexF64, m.physSpaces, [U1Space(0 => 1), U1Space(0 => 1)]; normalizeMPS = true)
        for dt in (0.05, 0.05im, 0.05 + 0.02im)
            evolved, _, _, _ = perform_timestep!(copy(p), H, dt, TDVP2(; extendBasis = false))
            reference = exp(-im * conj(dt) * matrix(H)) * vec(mps2vec(p))
            @test vec(mps2vec(evolved)) ≈ reference / norm(reference) atol = 1.0e-10
        end
        m = model(2)
        p = initializeVacuumMPS(m)
        H = 0 * HTTN.constructIdentityMPO(m.physSpaces, U1Space(0 => 1))
        for krylovDim in (0, 2)
            evolved, _, _, _ = perform_timestep!(copy(p), H, 0.05, TDVP2(; krylovDim))
            @test vec(mps2vec(evolved)) ≈ vec(mps2vec(p)) atol = 1.0e-12
        end
        H = HTTN.constructIdentityMPO(m.physSpaces, U1Space(0 => 1))
        for krylovDim in (0, 2)
            initial = copy(p)
            evolved, _, _, _ = perform_timestep!(initial, H, 0.05, TDVP2(; krylovDim))
            @test evolved === initial
            @test vec(mps2vec(evolved)) ≈ exp(-0.05im) * vec(mps2vec(p)) atol = 1.0e-12
        end
        @test applicable(perform_basisOptimization!, p, m, TDVP2BO())
    end
end
