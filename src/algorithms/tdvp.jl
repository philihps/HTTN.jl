
@kwdef struct TDVP2
    bondDim::Int64 = 1000
    krylovDim::Int = 2
    compressionAlg::String = "zipUp"
    truncErrT::Float64 = 1e-6
    truncErrK::Float64 = 1e-8
    truncErrM::Float64 = 1e-10
    doBasisExtend::Bool = true
    verbosePrint::Int64 = 0
end

@kwdef struct TDVP2BO
    bondDim::Int64 = 1000
    krylovDim::Int = 2
    compressionAlg::String = "zipUp"
    truncErrT::Float64 = 1e-6
    truncErrK::Float64 = 1e-8
    truncErrM::Float64 = 1e-10
    doBasisExtend::Bool = true
    verbosePrint::Int64 = 0
end

#-------------------------------------------------

function applyAC(x::TensorMap, mpo::TensorMap, envL::TensorMap, envR::TensorMap)
    @tensor x[-1 -2; -3] := envL[-1, 2, 1] * x[1, 3, 4] * mpo[2, -2, 5, 3] * envR[4, 5, -3]
    return x
end

function applyAC2(x::TensorMap, mpoL::TensorMap, mpoR::TensorMap, envL::TensorMap,
                  envR::TensorMap)
    @tensor x[-1 -2; -3 -4] := envL[-1, 2, 1] *
                               x[1, 3, 5, 6] *
                               mpoL[2, -2, 4, 3] *
                               mpoR[4, -3, 7, 5] *
                               envR[6, 7, -4]
end

#-------------------------------------------------

function extendMPS(finiteMPS::SparseMPS, krylovVectors::Vector{<:SparseMPS};
                   truncErrM::Float64 = 1e-6)
    """ Given an MPS |ψ(t)⟩ and a collection of Krylov vectors (H^l)|ψ(t)⟩, l = 1 : k, returns an MPS which is equal to |ψ(t)⟩ (has fidelity 1.0 with |ψ(t)⟩) but whose MPS basis
    is extended to contain a portion of the basis of the Krylov vectors, that is orthogonal to the MPS basis of |ψ(t)⟩ """

    # get length of finiteMPS
    N = length(finiteMPS)

    # combine |ψ(t)⟩ and Krylov vectors
    dimKrylovSpace = 1 + length(krylovVectors)
    vectorMPS = Vector{SparseMPS}(undef, 1 + length(krylovVectors))
    vectorMPS[1] = finiteMPS
    vectorMPS[2:end] = krylovVectors

    # bring into left-canonical form --> orthogonality center at site N
    for mpsIdx in 1:dimKrylovSpace
        orthogonalizeMPS!(vectorMPS[mpsIdx], N)
    end

    # enlarge physical space of |ψ(t)⟩ to include some basis of the Krylov vectors
    for siteIdx in N:-1:2

        # bring tensor into right-canonical form
        U, S, Vdag = tsvd(vectorMPS[1][siteIdx], ((1,), (2, 3)))
        @tensor nullSpaceProjector[-1 -2; -3 -4] := Vdag'[-1, -2, 1] * Vdag[1, -3, -4]
        nullSpaceProjector = one(nullSpaceProjector) - nullSpaceProjector

        # construct density matrix for Krylov vectors
        reducedDensityMatrix = zeros(ComplexF64, codomain(nullSpaceProjector),
                                     domain(nullSpaceProjector))
        for mpsIdx in 2:dimKrylovSpace
            mpsTensor = permute(vectorMPS[mpsIdx][siteIdx], ((1,), (2, 3)))
            @tensor rdm[-1 -2; -3 -4] := mpsTensor'[-1, -2, 1] * mpsTensor[1, -3, -4]
            reducedDensityMatrix += rdm
        end
        reducedDensityMatrix /= real(tr(reducedDensityMatrix))

        if norm(nullSpaceProjector) > 1e-8

            # normalize nullSpaceProjector
            nullSpaceProjector /= real(tr(nullSpaceProjector))

            # project reducedDensityMatrix by nullSpaceProjector
            reducedDensityMatrix = nullSpaceProjector * reducedDensityMatrix *
                                   nullSpaceProjector

            # diagonalize projected density matrix to compute UR', which spans part of right basis of the Krylov vectors, which is orthogonal to right basis of |ψ(t)⟩
            UR, SR, VR = tsvd(reducedDensityMatrix; trunc = truncerr(truncErrM),
                              alg = TensorKit.SVD())

            # # test if eigVecs is orthogonal to Vdag
            # overLap = UR' * Vdag';
            # normOverlap = norm(overLap);
            # display(normOverlap)
            # if normOverlap > 1E-10
            #     error("Non-zero overlap of extended basis with original basis")
            # end

            # form direct sum of Vdag and UR' over left virtual MPS tensor index
            isoLA = isometry(space(Vdag, 1) ⊕ space(UR', 1), space(Vdag, 1))
            isoLB = leftnull(isoLA)
            @tensor Bx[-1; -2 -3] := isoLA[-1, 1] * Vdag[1, -2, -3] +
                                     isoLB[-1, 1] * UR'[1, -2, -3]

        else
            Bx = Vdag
        end

        # permute physical index back
        Bx = permute(Bx, ((1, 2), (3,)))

        # shift orthogonality center one site to the left using Bx' and replace tensor at site siteIdx with Bx
        for mpsIdx in 1:dimKrylovSpace
            @tensor vectorMPS[mpsIdx][siteIdx - 1][-1 -2; -3] := vectorMPS[mpsIdx][siteIdx - 1][-1,
                                                                                                -2,
                                                                                                1] *
                                                                 vectorMPS[mpsIdx][siteIdx - 0][1,
                                                                                                3,
                                                                                                2] *
                                                                 conj(Bx[-3, 3, 2])
            vectorMPS[mpsIdx][siteIdx - 0] = Bx
        end
    end
    return vectorMPS[1]
end

function basisExtend(finiteMPS::SparseMPS, finiteMPO::SparseMPO, alg::Union{TDVP2,TDVP2BO})
    """ Function to extend the basis of |ψ⟩ by global Krylov vectors H|ψ⟩, (H^2)|ψ⟩, ..., (H^l)|ψ⟩ """

    # set maximal bond dimension of finiteMPS
    maxDimKrylovVectors = min(2 * maxLinkDimsMPS(finiteMPS), 3000)

    # construct Krylov vectors H|ψ⟩, (H^2)|ψ⟩, ..., (H^l)|ψ⟩
    krylovVectors = Vector{SparseMPS}(undef, alg.krylovDim)
    for idxK in 1:(alg.krylovDim)
        prevMPS = idxK == 1 ? finiteMPS : krylovVectors[idxK - 1]
        # krylovVectors[idxK] = normalizeMPS(applyMPO(finiteMPO, prevMPS, maxDim = maxDimKrylovVectors, truncErr = alg.truncErrK, compressionAlg = alg.compressionAlg));
        krylovVectors[idxK] = normalizeMPS(applyMPO(finiteMPO,
                                                    prevMPS;
                                                    truncErr = alg.truncErrK,
                                                    compressionAlg = alg.compressionAlg,))
        # println("maxLinkDim Krylov MPS : ", maxLinkDimsMPS(krylovVectors[idxK]))
    end

    # extend and return |ψ⟩
    finiteMPSX = extendMPS(finiteMPS, krylovVectors; truncErrM = alg.truncErrM)
    return finiteMPSX
end

#-------------------------------------------------

function perform_timestep!(finiteMPS::SparseMPS,
                           finiteMPO::SparseMPO,
                           timeStep::Union{Float64,ComplexF64},
                           alg::TDVP2)
    """ 2-site TDVP implementation for finiteMPO with global Krylov subspace expansion """

    if typeof(timeStep) == ComplexF64
        timeStep = -timeStep # t = -iτ
    end

    # make basis extension to include a number of global Krylov vectors
    if alg.doBasisExtend
        finiteMPS = basisExtend(finiteMPS, finiteMPO, alg)
    end

    # initialize MPO environments
    mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO)

    # initialize truncationErrors
    truncationErrors = Float64[]

    # sweep L ---> R
    for siteIdx in 1:+1:(length(finiteMPS) - 1)

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                      ((1, 2),
                       (3, 4)))

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        newAC2, convHist = exponentiate(x -> applyAC2(x,
                                                      finiteMPO[siteIdx + 0],
                                                      finiteMPO[siteIdx + 1],
                                                      mpoEnvL[siteIdx + 0],
                                                      mpoEnvR[siteIdx + 1]),
                                        -1im * timeStep / 2,
                                        AC2,
                                        Lanczos())

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2,
                          ((1, 2),
                           (3, 4));
                          trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT),
                          alg = TensorKit.SVD(),)
        S /= norm(S)
        U = permute(U, ((1, 2), (3,)))
        V = permute(S * V, ((1, 2), (3,)))
        truncationErrors = vcat(truncationErrors, ϵ)

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U
        finiteMPS[siteIdx + 1] = V

        # update mpoEnvL
        mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx],
                                              finiteMPO[siteIdx], finiteMPS[siteIdx])

        # compute K(n + 1) and apply it to V(n + 1) to evolve it with exp(+1im * timeStep/2 * K(n + 1))
        if siteIdx < (length(finiteMPS) - 1)
            finiteMPS[siteIdx + 1], convHist = exponentiate(x -> applyAC(x,
                                                                         finiteMPO[siteIdx + 1],
                                                                         mpoEnvL[siteIdx + 1],
                                                                         mpoEnvR[siteIdx + 1]),
                                                            +1im * timeStep / 2,
                                                            finiteMPS[siteIdx + 1],
                                                            Lanczos())
        end
    end

    # sweep L <--- R
    for siteIdx in (length(finiteMPS) - 1):-1:1

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                      ((1, 2),
                       (3, 4)))

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        newAC2, convHist = exponentiate(x -> applyAC2(x,
                                                      finiteMPO[siteIdx + 0],
                                                      finiteMPO[siteIdx + 1],
                                                      mpoEnvL[siteIdx + 0],
                                                      mpoEnvR[siteIdx + 1]),
                                        -1im * timeStep / 2,
                                        AC2,
                                        Lanczos())

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2,
                          ((1, 2),
                           (3, 4));
                          trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT),
                          alg = TensorKit.SVD(),)
        S /= norm(S)
        U = permute(U * S, ((1, 2), (3,)))
        V = permute(V, ((1, 2), (3,)))
        truncationErrors = vcat(truncationErrors, ϵ)

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U
        finiteMPS[siteIdx + 1] = V

        # update mpoEnvR
        mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                              finiteMPS[siteIdx + 1],
                                              finiteMPO[siteIdx + 1],
                                              finiteMPS[siteIdx + 1])

        # compute K(n) and apply it to V(n) to evolve it with exp(+1im * timeStep/2 * K(n))
        if siteIdx > 1
            finiteMPS[siteIdx + 0], convHist = exponentiate(x -> applyAC(x,
                                                                         finiteMPO[siteIdx + 0],
                                                                         mpoEnvL[siteIdx + 0],
                                                                         mpoEnvR[siteIdx + 0]),
                                                            +1im * timeStep / 2,
                                                            finiteMPS[siteIdx + 0],
                                                            Lanczos())
        end
    end

    # return optimized finiteMPS
    return finiteMPS, mpoEnvL, mpoEnvR, maximum(truncationErrors)
end

function perform_timestep(finiteMPS::SparseMPS,
                          finiteMPO::SparseMPO,
                          timeStep::Union{Float64,ComplexF64},
                          alg::TDVP2)
    return perform_timestep!(copy(finiteMPS), finiteMPO, timeStep, alg)
end

#-------------------------------------------------

function perform_timestep!(finiteMPS::SparseMPS,
                           mpoHandle::Function,
                           QFTModel::AbstractQFTModel,
                           timeStep::Union{Float64,ComplexF64},
                           alg::TDVP2BO)
    """ 2-site TDVP implementation for finiteMPO with global Krylov subspace expansion """

    # create MPO that should be simulated
    finiteMPO = mpoHandle(QFTModel)
    println(getLinkDimsMPO(finiteMPO))

    # get bogParameters
    bogParameters = QFTModel.modelParameters.truncationParameters[:bogParameters]
    println(bogParameters)

    # make basis extension to include a number of global Krylov vectors
    if alg.doBasisExtend
        finiteMPS = basisExtend(finiteMPS, finiteMPO, alg)
    end

    # initialize MPO environments
    mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO)

    # initialize truncationErrors
    truncationErrors = Float64[]

    # sweep L ---> R
    for siteIdx in 1:+1:(length(finiteMPS) - 1)

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                      ((1, 2),
                       (3, 4)))

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        newAC2, convHist = exponentiate(x -> applyAC2(x,
                                                      finiteMPO[siteIdx + 0],
                                                      finiteMPO[siteIdx + 1],
                                                      mpoEnvL[siteIdx + 0],
                                                      mpoEnvR[siteIdx + 1]),
                                        -1im * timeStep / 2,
                                        AC2,
                                        Lanczos())

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2,
                          ((1, 2),
                           (3, 4));
                          trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT),
                          alg = TensorKit.SVD(),)
        S /= norm(S)
        U = permute(U, ((1, 2), (3,)))
        V = permute(S * V, ((1, 2), (3,)))
        truncationErrors = vcat(truncationErrors, ϵ)

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U
        finiteMPS[siteIdx + 1] = V

        # update mpoEnvL
        mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx],
                                              finiteMPO[siteIdx], finiteMPS[siteIdx])

        # compute K(n + 1) and apply it to V(n + 1) to evolve it with exp(+1im * timeStep/2 * K(n + 1))
        if siteIdx < (length(finiteMPS) - 1)
            finiteMPS[siteIdx + 1], convHist = exponentiate(x -> applyAC(x,
                                                                         finiteMPO[siteIdx + 1],
                                                                         mpoEnvL[siteIdx + 1],
                                                                         mpoEnvR[siteIdx + 1]),
                                                            +1im * timeStep / 2,
                                                            finiteMPS[siteIdx + 1],
                                                            Lanczos())
        end
    end

    # sweep L <--- R
    for siteIdx in (length(finiteMPS) - 1):-1:1

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                      ((1, 2),
                       (3, 4)))

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        newAC2, convHist = exponentiate(x -> applyAC2(x,
                                                      finiteMPO[siteIdx + 0],
                                                      finiteMPO[siteIdx + 1],
                                                      mpoEnvL[siteIdx + 0],
                                                      mpoEnvR[siteIdx + 1]),
                                        -1im * timeStep / 2,
                                        AC2,
                                        Lanczos())

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2,
                          ((1, 2),
                           (3, 4));
                          trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT),
                          alg = TensorKit.SVD(),)
        S /= norm(S)
        U = permute(U * S, ((1, 2), (3,)))
        V = permute(V, ((1, 2), (3,)))
        truncationErrors = vcat(truncationErrors, ϵ)

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U
        finiteMPS[siteIdx + 1] = V

        # update mpoEnvR
        mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                              finiteMPS[siteIdx + 1],
                                              finiteMPO[siteIdx + 1],
                                              finiteMPS[siteIdx + 1])

        # compute K(n) and apply it to V(n) to evolve it with exp(+1im * timeStep/2 * K(n))
        if siteIdx > 1
            finiteMPS[siteIdx + 0], convHist = exponentiate(x -> applyAC(x,
                                                                         finiteMPO[siteIdx + 0],
                                                                         mpoEnvL[siteIdx + 0],
                                                                         mpoEnvR[siteIdx + 0]),
                                                            +1im * timeStep / 2,
                                                            finiteMPS[siteIdx + 0],
                                                            Lanczos())
        end
    end

    # # ------------------------------------------------------------
    # # perform local basis optimization to reduce entanglement

    # # sweep L ---> R
    # for siteIdx in 1:+1:(length(finiteMPS) - 1)

    #     # construct initial AC
    #     AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1,), (2, 3)), (1, 2), (3, 4))

    #     if mod(siteIdx, 2) == 0

    #         # get physVecSpaces for squeezing operator
    #         PL = space(finiteMPS[siteIdx + 0], 2)
    #         PR = space(finiteMPS[siteIdx + 1], 2)
    #         nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

    #         # set kL and kR
    #         kL = -1 * Int(siteIdx / 2)
    #         kR = +1 * Int(siteIdx / 2)
    #         # display([kL  kR])

    #         # compute cost function pre optimization
    #         costFuncPre = computeRenyiEntropy(AC2)
    #         # vNEntropyPre = computeEntropy(AC2);

    #         # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
    #         if alg.verbosePrint == 2
    #             listOfXiValues = collect(-0.6:0.05:+0.6)
    #             storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
    #             storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
    #             for (idx, ξ) in enumerate(listOfXiValues)
    #                 sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
    #                 storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
    #                                                                                                AC2))
    #                 # storeAnalyticGradient[idx] = analyticGradientCostFunction(ξ, nMax, kL,
    #                 #                                                           kR, PL, PR,
    #                 #                                                           AC2)
    #             end

    #             titleString = @sprintf("[k_L, k_R] = [%+d, %+d]", kL, kR)
    #             titleString = latexstring(titleString)
    #             renyiEntropyPlot = plot(listOfXiValues,
    #                                     storeEntanglementEntropy;
    #                                     linewidth = 2.0,
    #                                     xlabel = L"\xi",
    #                                     ylabel = L"S",
    #                                     label = L"S(\xi)",
    #                                     frame = :box,
    #                                     title = titleString,)
    #             # plot!(renyiEntropyPlot,
    #             #       listOfXiValues,
    #             #       storeAnalyticGradient;
    #             #       linewidth = 2.0,
    #             #       label = L"\partial S(\xi)/\partial \xi",)
    #             display(renyiEntropyPlot)
    #         end

    #         # # find optimimal ξ by roots of gradient
    #         # # optimalXi = find_zeros(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), -0.5, +0.5);
    #         # optimalXi = find_zero(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), 0.0)
    #         # @show optimalXi

    #         # vecξ = [real(bogParameters[kR]), imag(bogParameters[kR])]
    #         # fval, gval = value_and_gradient(vecξ, nMax, kL, kR, PL, PR, AC2)
    #         # println(fval)
    #         # println(gval)

    #         # optimize twoSiteUnitary
    #         # vecξ = [real(bogParameters[kR]), imag(bogParameters[kR])]
    #         optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, AC2),
    #                             bogParameters[kR],
    #                             LBFGS(12; verbosity = 1, maxiter = 25, gradtol = 1e-4), 
    #                             # scale! = _scale!, 
    #                             # add! = _add!, 
    #                             # inner = _inner, 
    #                             )
    #         optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes

    #         if any(abs.(optimalXi) .> 1e-3)

    #             # check acceptance of optimalXi
    #             newCostFunction = zeros(Float64, length(optimalXi))
    #             for (idx, ξ) in enumerate(optimalXi)

    #                 # transform AC2 with optimalS
    #                 optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
    #                 optimizedTheta = applyTwoModeTransformation(optimalS, AC2)

    #                 # compute cost function post optimiization
    #                 costFuncPost = computeRenyiEntropy(optimizedTheta)
    #                 # vNEntropyPost = computeEntropy(optimizedTheta);
    #                 display([costFuncPre costFuncPost costFuncPre > costFuncPost])
    #                 # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
    #                 # println()

    #                 newCostFunction[idx] = costFuncPost
    #             end

    #             # select optimalXi corresponding to lowest costFuncPost
    #             costFuncPost, minIdx = findmin(newCostFunction)
    #             optimalXi = optimalXi[minIdx]

    #             # decompose optimizedTheta
    #             if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi)

    #                 # update two site tensor
    #                 optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
    #                 AC2 = applyTwoModeTransformation(optimalS, AC2)
    #                 println("new optimal ξ = ", optimalXi)

    #                 # update QFTModel with new bogParameters
    #                 bogParameters[kR] += optimalXi
    #                 QFTModel = updateBogoliubovParameters(QFTModel, bogParameters)
    #                 println(bogParameters, "\n")

    #                 # recreate modified MPO
    #                 finiteMPO = mpoHandle(QFTModel)
    #             end
    #         end
    #     end

    #     # perform SVD and truncate to desired bond dimension (also move orthogonality center to the right)
    #     U, S, V, ϵ = tsvd(AC2,
    #             ((1, 2),
    #             (3, 4));
    #             trunc = truncdim(alg.bondDim) &
    #                     truncerr(alg.truncErrT),
    #             alg = TensorKit.SVD(),)
    #     S /= norm(S)
    #     U = permute(U, ((1, 2), (3,)))
    #     V = permute(S * V, ((1, 2), (3,)))
    #     truncationErrors = vcat(truncationErrors, ϵ)

    #     # assign updated tensors
    #     finiteMPS[siteIdx + 0] = U
    #     finiteMPS[siteIdx + 1] = V

    #     # update mpoEnvL
    #     mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx],
    #                                 finiteMPS[siteIdx],
    #                                 finiteMPO[siteIdx],
    #                                 finiteMPS[siteIdx])

    # end

    # # sweep L <--- R
    # for siteIdx in (length(finiteMPS) - 1):-1:1

    #     # construct initial AC
    #     AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1,), (2, 3)),
    #                   (1, 2), (3, 4))

    #     if mod(siteIdx, 2) == 0

    #         # get physVecSpaces for squeezing operator
    #         PL = space(finiteMPS[siteIdx + 0], 2)
    #         PR = space(finiteMPS[siteIdx + 1], 2)
    #         nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

    #         # set kL and kR
    #         kL = -1 * Int(siteIdx / 2)
    #         kR = +1 * Int(siteIdx / 2)
    #         # display([kL  kR])

    #         # compute cost function pre optimization
    #         costFuncPre = computeRenyiEntropy(AC2)
    #         # vNEntropyPre = computeEntropy(AC2);

    #         # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
    #         if alg.verbosePrint == 2
    #             listOfXiValues = collect(-0.6:0.05:+0.6)
    #             storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
    #             storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
    #             for (idx, ξ) in enumerate(listOfXiValues)
    #                 sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
    #                 storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
    #                                                                                                AC2))
    #                 # storeAnalyticGradient[idx] = analyticGradientCostFunction(ξ, nMax, kL,
    #                 #                                                           kR, PL, PR,
    #                 #                                                           AC2)
    #             end

    #             titleString = @sprintf("[k_L, k_R] = [%+d, %+d]", kL, kR)
    #             titleString = latexstring(titleString)
    #             renyiEntropyPlot = plot(listOfXiValues,
    #                                     storeEntanglementEntropy;
    #                                     linewidth = 2.0,
    #                                     xlabel = L"\xi",
    #                                     ylabel = L"S",
    #                                     label = L"S(\xi)",
    #                                     frame = :box,
    #                                     title = titleString,)
    #             # plot!(renyiEntropyPlot,
    #             #       listOfXiValues,
    #             #       storeAnalyticGradient;
    #             #       linewidth = 2.0,
    #             #       label = L"\partial S(\xi)/\partial \xi",)
    #             display(renyiEntropyPlot)
    #         end

    #         # # find optimimal ξ by roots of gradient
    #         # # optimalXi = find_zeros(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), -0.5, +0.5);
    #         # optimalXi = find_zero(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), 0.0)
    #         # @show optimalXi

    #         # optimize twoSiteUnitary
    #         # vecξ = [real(bogParameters[kR]), imag(bogParameters[kR])]
    #         optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, AC2),
    #                             bogParameters[kR],
    #                             LBFGS(12; verbosity = 1, maxiter = 25, gradtol = 1e-4), 
    #                             # scale! = _scale!, 
    #                             # add! = _add!, 
    #                             # inner = _inner, 
    #                             )
    #         optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes

    #         if any(abs.(optimalXi) .> 1e-3)

    #             # check acceptance of optimalXi
    #             newCostFunction = zeros(Float64, length(optimalXi))
    #             for (idx, ξ) in enumerate(optimalXi)

    #                 # transform AC2 with optimalS
    #                 optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
    #                 optimizedTheta = applyTwoModeTransformation(optimalS, AC2)

    #                 # compute cost function post optimiization
    #                 costFuncPost = computeRenyiEntropy(optimizedTheta)
    #                 # vNEntropyPost = computeEntropy(optimizedTheta);
    #                 display([costFuncPre costFuncPost costFuncPre > costFuncPost])
    #                 # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
    #                 # println()

    #                 newCostFunction[idx] = costFuncPost
    #             end

    #             # select optimalXi corresponding to lowest costFuncPost
    #             costFuncPost, minIdx = findmin(newCostFunction)
    #             optimalXi = optimalXi[minIdx]
    #             # @show optimalXi

    #             # decompose optimizedTheta
    #             if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi)

    #                 # update two site tensor
    #                 optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
    #                 AC2 = applyTwoModeTransformation(optimalS, AC2)
    #                 println("new optimal ξ = ", optimalXi)

    #                 # update QFTModel with new bogParameters
    #                 bogParameters[kR] += optimalXi
    #                 QFTModel = updateBogoliubovParameters(QFTModel, bogParameters)
    #                 println(bogParameters, "\n")

    #                 # recreate modified MPO
    #                 finiteMPO = mpoHandle(QFTModel)
    #             end
    #         end
    #     end

    #     #  perform SVD and truncate to desired bond dimension (also move orthogonality center to the left)
    #     U, S, V, ϵ = tsvd(AC2,
    #             ((1, 2),
    #             (3, 4));
    #             trunc = truncdim(alg.bondDim) &
    #                     truncerr(alg.truncErrT),
    #             alg = TensorKit.SVD(),)
    #     S /= norm(S)
    #     U = permute(U * S, ((1, 2), (3,)))
    #     V = permute(V, ((1, 2), (3,)))
    #     truncationErrors = vcat(truncationErrors, ϵ)

    #     # assign updated tensors
    #     finiteMPS[siteIdx + 0] = U
    #     finiteMPS[siteIdx + 1] = V

    #     # update mpoEnvR
    #     mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
    #                                 finiteMPS[siteIdx + 1],
    #                                 finiteMPO[siteIdx + 1],
    #                                 finiteMPS[siteIdx + 1])

    # end

    # return optimized finiteMPS
    return finiteMPS, mpoEnvL, mpoEnvR, bogParameters, maximum(truncationErrors), finiteMPO
end

function perform_timestep(finiteMPS::SparseMPS,
                          mpoHandle::Function,
                          QFTModel::AbstractQFTModel,
                          timeStep::Union{Float64,ComplexF64},
                          alg::TDVP2BO)
    return perform_timestep!(copy(finiteMPS), mpoHandle, QFTModel, timeStep, alg)
end



# ------------------------------------------------------------
# perform local basis optimization to reduce entanglement

function perform_basisOptimization!(finiteMPS::SparseMPS, QFTModel::AbstractQFTModel, alg::TDVP2)
    """ rotates pairs of modes [-k,+k] to optimal basis, such that the Renyi-1/2 entropy is minimized """

    # get bogParameters
    bogParameters = QFTModel.modelParameters.truncationParameters[:bogParameters]
    println(bogParameters)

    # # initialize MPO environments
    # mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO)

    # initialize truncationErrors
    truncationErrors = Float64[]

    # sweep L ---> R
    for siteIdx in 1:+1:(length(finiteMPS) - 1)

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1,), (2, 3)), (1, 2), (3, 4))

        if mod(siteIdx, 2) == 0

            # get physVecSpaces for squeezing operator
            PL = space(finiteMPS[siteIdx + 0], 2)
            PR = space(finiteMPS[siteIdx + 1], 2)
            nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

            # set kL and kR
            kL = -1 * Int(siteIdx / 2)
            kR = +1 * Int(siteIdx / 2)
            # display([kL  kR])

            # compute cost function pre optimization
            costFuncPre = computeRenyiEntropy(AC2)
            # vNEntropyPre = computeEntropy(AC2);

            # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
            if alg.verbosePrint == 2
                listOfXiValues = collect(-0.6:0.05:+0.6)
                storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
                storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
                for (idx, ξ) in enumerate(listOfXiValues)
                    sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                    storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
                                                                                                   AC2))
                    # storeAnalyticGradient[idx] = analyticGradientCostFunction(ξ, nMax, kL,
                    #                                                           kR, PL, PR,
                    #                                                           AC2)
                end

                titleString = @sprintf("[k_L, k_R] = [%+d, %+d]", kL, kR)
                titleString = latexstring(titleString)
                renyiEntropyPlot = plot(listOfXiValues,
                                        storeEntanglementEntropy;
                                        linewidth = 2.0,
                                        xlabel = L"\xi",
                                        ylabel = L"S",
                                        label = L"S(\xi)",
                                        frame = :box,
                                        title = titleString,)
                # plot!(renyiEntropyPlot,
                #       listOfXiValues,
                #       storeAnalyticGradient;
                #       linewidth = 2.0,
                #       label = L"\partial S(\xi)/\partial \xi",)
                display(renyiEntropyPlot)
            end

            # # find optimimal ξ by roots of gradient
            # # optimalXi = find_zeros(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), -0.5, +0.5);
            # optimalXi = find_zero(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), 0.0)
            # @show optimalXi

            
            # fval, gval = value_and_gradient(bogParameters[kR] + 0.05 * randn(eltype(bogParameters[kR])), nMax, kL, kR, PL, PR, AC2)
            # println(fval)
            # println(gval)

            # optimize twoSiteUnitary
            # vecξ = [real(bogParameters[kR]), imag(bogParameters[kR])]
            optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, AC2),
                                bogParameters[kR] + 0.05 * randn(eltype(bogParameters[kR])),
                                LBFGS(12; verbosity = 1, maxiter = 50, gradtol = 1e-4), 
                                # scale! = _scale!, 
                                # add! = _add!, 
                                # inner = _inner, 
                                )
            optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes

            if any(abs.(optimalXi) .> 1e-3)

                # check acceptance of optimalXi
                newCostFunction = zeros(Float64, length(optimalXi))
                for (idx, ξ) in enumerate(optimalXi)

                    # transform AC2 with optimalS
                    optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                    optimizedTheta = applyTwoModeTransformation(optimalS, AC2)

                    # compute cost function post optimiization
                    costFuncPost = computeRenyiEntropy(optimizedTheta)
                    # vNEntropyPost = computeEntropy(optimizedTheta);
                    display([costFuncPre costFuncPost costFuncPre > costFuncPost])
                    # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
                    # println()

                    newCostFunction[idx] = costFuncPost
                end

                # select optimalXi corresponding to lowest costFuncPost
                costFuncPost, minIdx = findmin(newCostFunction)
                optimalXi = optimalXi[minIdx]

                # decompose optimizedTheta
                if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi)

                    # update two site tensor
                    optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
                    AC2 = applyTwoModeTransformation(optimalS, AC2)
                    println("new optimal ξ = ", optimalXi)

                    # update QFTModel with new bogParameters
                    bogParameters[kR] += optimalXi
                    QFTModel = updateBogoliubovParameters(QFTModel, bogParameters)
                    println(bogParameters, "\n")

                    # # recreate modified MPO
                    # finiteMPO = mpoHandle(QFTModel)
                end
            end
        end

        # perform SVD and truncate to desired bond dimension (also move orthogonality center to the right)
        U, S, V, ϵ = tsvd(AC2,
                ((1, 2),
                (3, 4));
                trunc = truncdim(alg.bondDim) &
                        truncerr(alg.truncErrT),
                alg = TensorKit.SVD(),)
        S /= norm(S)
        U = permute(U, ((1, 2), (3,)))
        V = permute(S * V, ((1, 2), (3,)))
        truncationErrors = vcat(truncationErrors, ϵ)

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U
        finiteMPS[siteIdx + 1] = V

        # # update mpoEnvL
        # mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx],
        #                             finiteMPS[siteIdx],
        #                             finiteMPO[siteIdx],
        #                             finiteMPS[siteIdx])

    end

    # sweep L <--- R
    for siteIdx in (length(finiteMPS) - 1):-1:1

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1,), (2, 3)), (1, 2), (3, 4))

        if mod(siteIdx, 2) == 0

            # get physVecSpaces for squeezing operator
            PL = space(finiteMPS[siteIdx + 0], 2)
            PR = space(finiteMPS[siteIdx + 1], 2)
            nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

            # set kL and kR
            kL = -1 * Int(siteIdx / 2)
            kR = +1 * Int(siteIdx / 2)
            # display([kL  kR])

            # compute cost function pre optimization
            costFuncPre = computeRenyiEntropy(AC2)
            # vNEntropyPre = computeEntropy(AC2);

            # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
            if alg.verbosePrint == 2
                listOfXiValues = collect(-0.6:0.05:+0.6)
                storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
                storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
                for (idx, ξ) in enumerate(listOfXiValues)
                    sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                    storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
                                                                                                   AC2))
                    # storeAnalyticGradient[idx] = analyticGradientCostFunction(ξ, nMax, kL,
                    #                                                           kR, PL, PR,
                    #                                                           AC2)
                end

                titleString = @sprintf("[k_L, k_R] = [%+d, %+d]", kL, kR)
                titleString = latexstring(titleString)
                renyiEntropyPlot = plot(listOfXiValues,
                                        storeEntanglementEntropy;
                                        linewidth = 2.0,
                                        xlabel = L"\xi",
                                        ylabel = L"S",
                                        label = L"S(\xi)",
                                        frame = :box,
                                        title = titleString,)
                # plot!(renyiEntropyPlot,
                #       listOfXiValues,
                #       storeAnalyticGradient;
                #       linewidth = 2.0,
                #       label = L"\partial S(\xi)/\partial \xi",)
                display(renyiEntropyPlot)
            end

            # # find optimimal ξ by roots of gradient
            # # optimalXi = find_zeros(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), -0.5, +0.5);
            # optimalXi = find_zero(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, AC2), 0.0)
            # @show optimalXi

            # optimize twoSiteUnitary
            # vecξ = [real(bogParameters[kR]), imag(bogParameters[kR])]
            optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, AC2),
                                bogParameters[kR] + 0.05 * randn(eltype(bogParameters[kR])),
                                LBFGS(12; verbosity = 1, maxiter = 50, gradtol = 1e-4), 
                                # scale! = _scale!, 
                                # add! = _add!, 
                                # inner = _inner, 
                                )
            optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes

            if any(abs.(optimalXi) .> 1e-3)

                # check acceptance of optimalXi
                newCostFunction = zeros(Float64, length(optimalXi))
                for (idx, ξ) in enumerate(optimalXi)

                    # transform AC2 with optimalS
                    optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                    optimizedTheta = applyTwoModeTransformation(optimalS, AC2)

                    # compute cost function post optimiization
                    costFuncPost = computeRenyiEntropy(optimizedTheta)
                    # vNEntropyPost = computeEntropy(optimizedTheta);
                    display([costFuncPre costFuncPost costFuncPre > costFuncPost])
                    # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
                    # println()

                    newCostFunction[idx] = costFuncPost
                end

                # select optimalXi corresponding to lowest costFuncPost
                costFuncPost, minIdx = findmin(newCostFunction)
                optimalXi = optimalXi[minIdx]
                # @show optimalXi

                # decompose optimizedTheta
                if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR], optimalXi)

                    # update two site tensor
                    optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
                    AC2 = applyTwoModeTransformation(optimalS, AC2)
                    println("new optimal ξ = ", optimalXi)

                    # update QFTModel with new bogParameters
                    bogParameters[kR] += optimalXi
                    QFTModel = updateBogoliubovParameters(QFTModel, bogParameters)
                    println(bogParameters, "\n")

                    # # recreate modified MPO
                    # finiteMPO = mpoHandle(QFTModel)
                end
            end
        end

        #  perform SVD and truncate to desired bond dimension (also move orthogonality center to the left)
        U, S, V, ϵ = tsvd(AC2,
                ((1, 2),
                (3, 4));
                trunc = truncdim(alg.bondDim) &
                        truncerr(alg.truncErrT),
                alg = TensorKit.SVD(),)
        S /= norm(S)
        U = permute(U * S, ((1, 2), (3,)))
        V = permute(V, ((1, 2), (3,)))
        truncationErrors = vcat(truncationErrors, ϵ)

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U
        finiteMPS[siteIdx + 1] = V

        # # update mpoEnvR
        # mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
        #                             finiteMPS[siteIdx + 1],
        #                             finiteMPO[siteIdx + 1],
        #                             finiteMPS[siteIdx + 1])

    end

    # return optimized finiteMPS
    return finiteMPS, bogParameters, truncationErrors

end