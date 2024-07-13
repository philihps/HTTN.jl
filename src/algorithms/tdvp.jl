
@kwdef struct TDVP2
    bondDim::Int64 = 10
    krylovDim::Int = 2
    compressionAlg::String = "zipUp"
    truncErrT::Float64 = 1e-6
    truncErrK::Float64 = 1e-8
    truncErrM::Float64 = 1e-10
    doBasisExtend::Bool = true
    verbosePrint = false
end

function perform_timestep!(finiteMPS::SparseMPS, finiteMPO::SparseMPO, timeStep::Union{Float64, ComplexF64}, alg::TDVP2)
    """ 2-site TDVP implementation for finiteMPO with global Krylov subspace expansion """

    # make basis extension to include a number of global Krylov vectors
    if alg.doBasisExtend
        finiteMPS = basisExtend(finiteMPS, finiteMPO, alg);
    end

    # initialize MPO environments
    mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO);

    # initialize truncationErrors
    truncationErrors = Float64[];

    # sweep L ---> R
    for siteIdx = 1 : +1 : (length(finiteMPS) - 1)

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        newAC2, convHist = exponentiate(x -> applyAC2(x, finiteMPO[siteIdx + 0], finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 1]), -1im * timeStep / 2, AC2, Lanczos());

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT), alg = TensorKit.SVD());
        S /= norm(S);
        U = permute(U, (1, 2), (3, ));
        V = permute(S * V, (1, 2), (3, ));
        truncationErrors = vcat(truncationErrors, ϵ);

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U;
        finiteMPS[siteIdx + 1] = V;

        # update mpoEnvL
        mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

        # compute K(n + 1) and apply it to V(n + 1) to evolve it with exp(+1im * timeStep/2 * K(n + 1))
        if siteIdx < (length(finiteMPS) - 1)
            finiteMPS[siteIdx + 1], convHist = exponentiate(x -> applyAC(x, finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 1], mpoEnvR[siteIdx + 1]), +1im * timeStep / 2, finiteMPS[siteIdx + 1], Lanczos());
        end

    end

    # sweep L <--- R
    for siteIdx = (length(finiteMPS) - 1) : -1 : 1

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        newAC2, convHist = exponentiate(x -> applyAC2(x, finiteMPO[siteIdx + 0], finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 1]), -1im * timeStep / 2, AC2, Lanczos());

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT), alg = TensorKit.SVD());
        S /= norm(S);
        U = permute(U * S, (1, 2), (3, ));
        V = permute(V, (1, 2), (3, ));
        truncationErrors = vcat(truncationErrors, ϵ);

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U;
        finiteMPS[siteIdx + 1] = V;

        # update mpoEnvR
        mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1], finiteMPS[siteIdx + 1], finiteMPO[siteIdx + 1], finiteMPS[siteIdx + 1]);

        # compute K(n) and apply it to V(n) to evolve it with exp(+1im * timeStep/2 * K(n))
        if siteIdx > 1
            finiteMPS[siteIdx + 0], convHist = exponentiate(x -> applyAC(x, finiteMPO[siteIdx + 0], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 0]), +1im * timeStep / 2, finiteMPS[siteIdx + 0], Lanczos());
        end

    end

    # return optimized finiteMPS
    return finiteMPS, mpoEnvL, mpoEnvR, maximum(truncationErrors);

end

function perform_timestep(finiteMPS::SparseMPS, finiteMPO::SparseMPO, timeStep::Union{Float64, ComplexF64}, alg::TDVP2)
    return perform_timestep!(copy(finiteMPS), finiteMPO, timeStep, alg)
end

#-------------------------------------------------

function applyAC(x::TensorMap, mpo::TensorMap, envL::TensorMap, envR::TensorMap)
    @tensor x[-1 -2; -3] := envL[-1, 2, 1] * x[1, 3, 4] * mpo[2, -2, 5, 3] * envR[4, 5, -3]
    return x;
end

function applyAC2(x::TensorMap, mpoL::TensorMap, mpoR::TensorMap, envL::TensorMap, envR::TensorMap)
    @tensor x[-1 -2; -3 -4] := envL[-1, 2, 1] * x[1, 3, 5, 6] * mpoL[2, -2, 4, 3] * mpoR[4, -3, 7, 5] * envR[6, 7, -4];
end

#-------------------------------------------------

function extendMPS(finiteMPS::SparseMPS, krylovVectors::Vector{<:SparseMPS}; truncErrM::Float64 = 1e-6)
    """ Given an MPS |ψ(t)⟩ and a collection of Krylov vectors (H^l)|ψ(t)⟩, l = 1 : k, returns an MPS which is equal to |ψ(t)⟩ (has fidelity 1.0 with |ψ(t)⟩) but whose MPS basis
    is extended to contain a portion of the basis of the Krylov vectors, that is orthogonal to the MPS basis of |ψ(t)⟩ """

    # get length of finiteMPS
    N = length(finiteMPS);

    # combine |ψ(t)⟩ and Krylov vectors
    dimKrylovSpace = 1 + length(krylovVectors);
    vectorMPS = Vector{SparseMPS}(undef, 1 + length(krylovVectors));
    vectorMPS[1] = finiteMPS;
    vectorMPS[2 : end] = krylovVectors;

    # bring into left-canonical form --> orthogonality center at site N
    for mpsIdx = 1 : dimKrylovSpace
        orthogonalizeMPS!(vectorMPS[mpsIdx], N);
    end

    # enlarge physical space of |ψ(t)⟩ to include some basis of the Krylov vectors
    for siteIdx = N : -1 : 2

        # bring tensor into right-canonical form
        U, S, Vdag = tsvd(vectorMPS[1][siteIdx], (1, ), (2, 3));
        @tensor nullSpaceProjector[-1 -2; -3 -4] := Vdag'[-1, -2, 1] * Vdag[1, -3, -4];
        nullSpaceProjector = one(nullSpaceProjector) - nullSpaceProjector;

        # construct density matrix for Krylov vectors
        reducedDensityMatrix = TensorMap(zeros, ComplexF64, codomain(nullSpaceProjector), domain(nullSpaceProjector));
        for mpsIdx = 2 : dimKrylovSpace
            mpsTensor = permute(vectorMPS[mpsIdx][siteIdx], (1, ), (2, 3));
            @tensor rdm[-1 -2; -3 -4] := mpsTensor'[-1, -2, 1] * mpsTensor[1, -3, -4];
            reducedDensityMatrix += rdm;
        end
        reducedDensityMatrix /= real(tr(reducedDensityMatrix));

        if norm(nullSpaceProjector) > 1e-8

            # normalize nullSpaceProjector
            nullSpaceProjector /= real(tr(nullSpaceProjector));

            # project reducedDensityMatrix by nullSpaceProjector
            reducedDensityMatrix = nullSpaceProjector * reducedDensityMatrix * nullSpaceProjector;

            # diagonalize projected density matrix to compute UR', which spans part of right basis of the Krylov vectors, which is orthogonal to right basis of |ψ(t)⟩
            UR, SR, VR = tsvd(reducedDensityMatrix, trunc = truncerr(truncErrM), alg = TensorKit.SVD());

            # # test if eigVecs is orthogonal to Vdag
            # overLap = UR' * Vdag';
            # normOverlap = norm(overLap);
            # display(normOverlap)
            # if normOverlap > 1E-10
            #     error("Non-zero overlap of extended basis with original basis")
            # end

            # form direct sum of Vdag and UR' over left virtual MPS tensor index
            isoLA = isometry(space(Vdag, 1) ⊕ space(UR', 1), space(Vdag, 1));
            isoLB = leftnull(isoLA);
            @tensor Bx[-1; -2 -3] := isoLA[-1, 1] * Vdag[1, -2, -3] + isoLB[-1, 1] * UR'[1, -2, -3];

        else
            Bx = Vdag;
        end

        # permute physical index back
        Bx = permute(Bx, (1, 2), (3, ));

        # shift orthogonality center one site to the left using Bx' and replace tensor at site siteIdx with Bx
        for mpsIdx = 1 : dimKrylovSpace
            @tensor vectorMPS[mpsIdx][siteIdx - 1][-1 -2; -3] := vectorMPS[mpsIdx][siteIdx - 1][-1, -2, 1] * vectorMPS[mpsIdx][siteIdx - 0][1, 3, 2] * conj(Bx[-3, 3, 2]);
            vectorMPS[mpsIdx][siteIdx - 0] = Bx;
        end

    end
    return vectorMPS[1];

end

function basisExtend(finiteMPS::SparseMPS, finiteMPO::SparseMPO, alg::TDVP2)
    """ Function to extend the basis of |ψ⟩ by global Krylov vectors H|ψ⟩, (H^2)|ψ⟩, ..., (H^l)|ψ⟩ """

    # set maximal bond dimension of finiteMPS
    maxDimKrylovVectors = min(2 * maxLinkDimsMPS(finiteMPS), 3000);

    # construct Krylov vectors H|ψ⟩, (H^2)|ψ⟩, ..., (H^l)|ψ⟩
    krylovVectors = Vector{SparseMPS}(undef, alg.krylovDim);
    for idxK in 1 : alg.krylovDim
        prevMPS = idxK == 1 ? finiteMPS : krylovVectors[idxK - 1];
        # krylovVectors[idxK] = normalizeMPS(applyMPO(finiteMPO, prevMPS, maxDim = maxDimKrylovVectors, truncErr = alg.truncErrK, compressionAlg = alg.compressionAlg));
        krylovVectors[idxK] = normalizeMPS(applyMPO(finiteMPO, prevMPS, truncErr = alg.truncErrK, compressionAlg = alg.compressionAlg));
        # println("maxLinkDim Krylov MPS : ", maxLinkDimsMPS(krylovVectors[idxK]))
    end

    # extend and return |ψ⟩
    finiteMPSX = extendMPS(finiteMPS, krylovVectors, truncErrM = alg.truncErrM);
    return finiteMPSX;

end