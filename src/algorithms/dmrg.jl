""" two-site DMRG algorithm """

@kwdef struct DMRG2
    bondDim::Int64 = 10
    truncErr::Float64 = 1e-6
    convTolE::Float64 = 1e-6
    eigsTol::Float64 = 1e-8
    maxIterationsInit::Int64 = 10
    maxIterations::Int64 = 1
    subspaceExpansion::Bool = true
    verbosePrint::Int64 = 0
end

@kwdef struct DMRG2BO
    bondDim::Int64 = 10
    truncErr::Float64 = 1e-6
    convTolE::Float64 = 1e-6
    eigsTol::Float64 = 1e-8
    maxIterationsInit::Int64 = 10
    maxIterations::Int64 = 5
    subspaceExpansion::Bool = true
    startOptimization::Int = 1
    verbosePrint::Int64 = 0
end

function applyH2(X, EL, mpo1, mpo2, ER)
    @tensor X[-1 -2; -3 -4] := EL[-1, 2, 1] * X[1, 3, 5, 6] * mpo1[2, -2, 4, 3] *
                               mpo2[4, -3, 7, 5] * ER[6, 7, -4]
    return X
end

function applyH2_X(X, siteIdx, EL, MPO, ER, PLs, Y, PRs, penaltyWeights::Vector{Float64})

    # compute H|ψexcited⟩
    @tensor X1[-1 -2; -3 -4] := EL[siteIdx][-1, 2, 1] *
                                X[1, 3, 5, 6] *
                                MPO[siteIdx][2, -2, 4, 3] *
                                MPO[siteIdx + 1][4, -3, 7, 5] *
                                ER[siteIdx + 1][6, 7, -4]

    # compute overlaps ⟨ψ0|ψexcited⟩, ⟨ψ1|ψexcited⟩, ...
    waveFunctionOverlaps = Vector{Union{Float64,ComplexF64}}(undef, length(Y))
    for orthIdx in eachindex(Y)
        waveFunctionOverlap = @tensor PLs[orthIdx][siteIdx][1, 2] *
                                      X[2, 3, 4, 5] *
                                      conj(Y[orthIdx][1, 3, 4, 6]) *
                                      PRs[orthIdx][siteIdx + 1][5, 6]
        waveFunctionOverlaps[orthIdx] = waveFunctionOverlap
    end

    # compute ω0 P0|ψexcited⟩, ω1 P1|ψexcited⟩, ... 
    projOverlaps = Vector{TensorMap{ComplexF64}}(undef, length(Y))
    for orthIdx in eachindex(Y)
        @tensor XO[-1 -2; -3 -4] := conj(PLs[orthIdx][siteIdx][1, -1]) *
                                    Y[orthIdx][1, -2, -3, 4] *
                                    conj(PRs[orthIdx][siteIdx + 1][-4, 4])
        projOverlaps[orthIdx] = penaltyWeights[orthIdx] * waveFunctionOverlaps[orthIdx] * XO
    end

    # return sum of terms
    return X1 + sum(projOverlaps)
end

function find_groundstate!(finiteMPS::SparseMPS, finiteMPO::SparseMPO, alg::DMRG2)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                             compressionAlg = "zipUp")
    end

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0]

    # store vector of truncation errors
    ϵs = ones(Float64, length(finiteMPS) - 1)

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1
    maxOptimSteps = 1
    runOptimization = true
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO)

        # initialize vector to store energies
        oldEigsEnergies = zeros(Float64, length(finiteMPS) - 1)

        # store vector of truncation errors
        ϵs = ones(Float64, length(finiteMPS) - 1)

        # main DMRG loop
        loopCounter = 1
        runOptimizationDMRG = true
        while runOptimizationDMRG

            # initialize vector to store energies
            newEigsEnergies = zeros(Float64, length(finiteMPS) - 1)

            # sweep L ---> R
            for siteIdx in 1:+1:(length(finiteMPS) - 1)

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] *
                                permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                ((1, 2),
                                 (3, 4)))

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(theta,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2(x,
                                   mpoEnvL[siteIdx],
                                   finiteMPO[siteIdx],
                                   finiteMPO[siteIdx + 1],
                                   mpoEnvR[siteIdx + 1])
                end
                eigVal = eigenVal[1]
                newTheta = eigenVec[1]
                newEigsEnergies[siteIdx] = eigVal

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U, ((1, 2), (3,)))
                V = permute(S * V, ((1, 2), (3,)))

                # compute error
                v = @tensor theta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx],
                                                      finiteMPS[siteIdx],
                                                      finiteMPO[siteIdx],
                                                      finiteMPS[siteIdx])
            end

            # sweep L <--- R
            for siteIdx in (length(finiteMPS) - 1):-1:1

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] *
                                permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                ((1, 2),
                                 (3, 4)))

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(theta,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2(x,
                                   mpoEnvL[siteIdx],
                                   finiteMPO[siteIdx],
                                   finiteMPO[siteIdx + 1],
                                   mpoEnvR[siteIdx + 1])
                end
                eigVal = eigenVal[1]
                newTheta = eigenVec[1]
                newEigsEnergies[siteIdx] = eigVal

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U * S, ((1, 2), (3,)))
                V = permute(V, ((1, 2), (3,)))

                # compute error
                v = @tensor theta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1],
                                                      finiteMPO[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1])
            end

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO)
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal)
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal)

            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end])
            # energyConvergenceA = norm(newEigsEnergies .- oldEigsEnergies) / length(finiteMPS);
            # display([energyConvergence energyConvergenceA maximum(ϵs)])
            alg.verbosePrint > 0 &&
                @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n",
                        loopCounter,
                        mpoExpVal,
                        energyConvergence)
            if energyConvergence < alg.convTolE
                runOptimizationDMRG = false
            end

            # increase loopCounter and update oldEigsEnergies
            loopCounter += 1
            # oldEigsEnergies = copy(newEigsEnergies);

        end

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO)
        @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance)

        # re-randomize finiteMPS if non-eigenstate was found
        if energyVariance > 1e-0
            if optimizationLoopCounter < maxOptimSteps
                @printf("\nre-randomizing MPS...\n")
                for idxMPS in eachindex(finiteMPS)
                    finiteMPS[idxMPS] += 0.1 *
                                         randn(ComplexF64, codomain(finiteMPS[idxMPS]),
                                               domain(finiteMPS[idxMPS]))
                end
                finiteMPS = normalizeMPS(finiteMPS)
                finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                                     compressionAlg = "zipUp")
            else
                runOptimization = false
            end
        else
            runOptimization = false
        end
        # runOptimization = false;

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1
    end
    # @printf("\n")

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end]
    return finiteMPS, finalEnergy, ϵs
end

# # LineSearch settings
# optimAlg = LBFGS(8, verbosity = 1, maxiter  = 25, gradtol = 1e-6);

# function to check acceptance of new basis
function checkAcceptance(oldVal::Float64, newVal::Float64, oldXi::Number, newXi::Number)
    if (oldVal - newVal > 1e-6)
        return true
    else
        return false
    end
end

function find_groundstate!(finiteMPS::SparseMPS, mpoHandle::Function,
                           QFTModel::AbstractQFTModel, alg::DMRG2BO)

    # create MPO that should be simulated
    finiteMPO = mpoHandle(QFTModel)
    println(getLinkDimsMPO(finiteMPO))

    # get bogParameters
    bogParameters = QFTModel.modelParameters.truncationParameters[:bogParameters]
    println(bogParameters)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                             compressionAlg = "zipUp")
    end

    # initialize array to store Bogoliubov parameters
    storeBogoliubovParameters = zeros(Float64, 0, length(bogParameters))
    # storeBogoliubovParameters = vcat(storeBogoliubovParameters, reshape(bogParameters, 1, length(bogParameters)));

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0]

    # initialize array to store truncation errors
    ϵs = zeros(Float64, length(finiteMPS) - 1)

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1
    maxOptimSteps = 1
    runOptimization = true
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO)

        # initialize vector to store energies
        oldEigsEnergies = zeros(Float64, length(finiteMPS) - 1, 2)

        # store vector of truncation errors
        ϵs = ones(Float64, length(finiteMPS) - 1)

        # main DMRG loop
        loopCounter = 1
        runOptimizationDMRG = true
        while runOptimizationDMRG

            # initialize vector to store energies
            newEigsEnergies = zeros(Float64, length(finiteMPS) - 1, 2)

            # sweep L ---> R
            for siteIdx in 1:+1:(length(finiteMPS) - 1)

                # @info "optimizing site" siteIdx

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] *
                                permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                ((1, 2),
                                 (3, 4)))

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(theta,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2(x,
                                   mpoEnvL[siteIdx],
                                   finiteMPO[siteIdx],
                                   finiteMPO[siteIdx + 1],
                                   mpoEnvR[siteIdx + 1])
                end
                eigVal = eigenVal[1]
                newTheta = eigenVec[1]
                newEigsEnergies[siteIdx, 1] = eigVal

                # ------------------------------------------------------------
                # perform local basis optimization to reduce entanglement

                if mod(siteIdx, 2) == 0 && loopCounter > alg.startOptimization

                    # get physVecSpaces for squeezing operator
                    PL = space(finiteMPS[siteIdx + 0], 2)
                    PR = space(finiteMPS[siteIdx + 1], 2)
                    nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

                    # set kL and kR
                    kL = -1 * Int(siteIdx / 2)
                    kR = +1 * Int(siteIdx / 2)
                    # display([kL  kR])

                    # compute cost function pre optimization
                    costFuncPre = computeRenyiEntropy(newTheta)
                    vNEntropyPre = computeEntropy(newTheta)

                    # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
                    if alg.verbosePrint == 2
                        listOfXiValues = collect(-0.4:0.02:+0.4)
                        storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
                        storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
                        for (idx, ξ) in enumerate(listOfXiValues)
                            sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
                                                                                                           newTheta))
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
                        display(renyiEntropyPlot)
                    end

                    # optimize twoSiteUnitary
                    optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR,
                                                                newTheta),
                                        bogParameters[1 + kR] +
                                        0.05 * randn(eltype(bogParameters[1 + kR])),
                                        LBFGS(12; verbosity = 1, maxiter = 50,
                                              gradtol = 1e-4))
                    optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes

                    if any(abs.(optimalXi) .> 1e-3)

                        # check acceptance of optimalXi
                        newCostFunction = zeros(Float64, length(optimalXi))
                        for (idx, ξ) in enumerate(optimalXi)

                            # transform newTheta with optimalS
                            optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            optimizedTheta = applyTwoModeTransformation(optimalS, newTheta)

                            # compute cost function post optimiization
                            costFuncPost = computeRenyiEntropy(optimizedTheta)
                            vNEntropyPost = computeEntropy(optimizedTheta)
                            # display([costFuncPre costFuncPost costFuncPre > costFuncPost])
                            # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
                            # println()

                            newCostFunction[idx] = costFuncPost
                        end

                        # select optimalXi corresponding to lowest costFuncPost
                        costFuncPost, minIdx = findmin(newCostFunction)
                        optimalXi = optimalXi[minIdx]
                        # @show optimalXi

                        # decompose optimizedTheta
                        if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1],
                                           optimalXi)

                            # update two site tensor
                            optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
                            newTheta = applyTwoModeTransformation(optimalS, newTheta)
                            println("new optimal ξ = ", optimalXi)

                            # update QFTModel with new bogParameters
                            bogParameters[1 + kR] += optimalXi
                            QFTModel = updateBogoliubovParameters(QFTModel;
                                                                  bogoliubovRot = true,
                                                                  bogParameters = bogParameters)
                            println(bogParameters, "\n")

                            # recreate modified MPO
                            finiteMPO = mpoHandle(QFTModel)
                        end
                    end
                end

                # ------------------------------------------------------------

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U, ((1, 2), (3,)))
                V = permute(S * V, ((1, 2), (3,)))

                # compute error
                v = @tensor theta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx],
                                                      finiteMPS[siteIdx],
                                                      finiteMPO[siteIdx],
                                                      finiteMPS[siteIdx])
            end

            # # store bogParameters
            # storeBogoliubovParameters = vcat(storeBogoliubovParameters, reshape(bogParameters, 1, length(bogParameters)));

            # sweep L <--- R
            for siteIdx in (length(finiteMPS) - 1):-1:1

                # @info "optimizing site" siteIdx

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] *
                                permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                ((1, 2),
                                 (3, 4)))

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(theta,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2(x,
                                   mpoEnvL[siteIdx],
                                   finiteMPO[siteIdx],
                                   finiteMPO[siteIdx + 1],
                                   mpoEnvR[siteIdx + 1])
                end
                eigVal = eigenVal[1]
                newTheta = eigenVec[1]
                newEigsEnergies[siteIdx, 2] = eigVal

                # ------------------------------------------------------------
                # perform local basis optimization to reduce entanglement

                if mod(siteIdx, 2) == 0 && loopCounter > alg.startOptimization

                    # get physVecSpaces for squeezing operator
                    PL = space(finiteMPS[siteIdx + 0], 2)
                    PR = space(finiteMPS[siteIdx + 1], 2)
                    nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

                    # set kL and kR
                    kL = -1 * Int(siteIdx / 2)
                    kR = +1 * Int(siteIdx / 2)
                    # display([kL  kR])

                    # compute cost function pre optimization
                    costFuncPre = computeRenyiEntropy(newTheta)
                    vNEntropyPre = computeEntropy(newTheta)

                    # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
                    if alg.verbosePrint == 2
                        listOfXiValues = collect(-0.4:0.02:+0.4)
                        storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
                        storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
                        for (idx, ξ) in enumerate(listOfXiValues)
                            sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
                                                                                                           newTheta))
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
                        display(renyiEntropyPlot)
                    end

                    # optimize twoSiteUnitary
                    optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR,
                                                                newTheta),
                                        bogParameters[1 + kR] +
                                        0.05 * randn(eltype(bogParameters[1 + kR])),
                                        LBFGS(12; verbosity = 1, maxiter = 50,
                                              gradtol = 1e-4))
                    optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes

                    if any(abs.(optimalXi) .> 1e-3)

                        # check acceptance of optimalXi
                        newCostFunction = zeros(Float64, length(optimalXi))
                        for (idx, ξ) in enumerate(optimalXi)

                            # transform newTheta with optimalS
                            optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            optimizedTheta = applyTwoModeTransformation(optimalS, newTheta)

                            # compute cost function post optimiization
                            costFuncPost = computeRenyiEntropy(optimizedTheta)
                            vNEntropyPost = computeEntropy(optimizedTheta)
                            # display([costFuncPre costFuncPost costFuncPre > costFuncPost])
                            # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
                            # println()

                            newCostFunction[idx] = costFuncPost
                        end

                        # select optimalXi corresponding to lowest costFuncPost
                        costFuncPost, minIdx = findmin(newCostFunction)
                        optimalXi = optimalXi[minIdx]
                        # @show optimalXi

                        # decompose optimizedTheta
                        if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1],
                                           optimalXi)

                            # update two site tensor
                            optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
                            newTheta = applyTwoModeTransformation(optimalS, newTheta)
                            println("new optimal ξ = ", optimalXi)

                            # update QFTModel with new bogParameters
                            bogParameters[1 + kR] += optimalXi
                            QFTModel = updateBogoliubovParameters(QFTModel;
                                                                  bogoliubovRot = true,
                                                                  bogParameters = bogParameters)
                            println(bogParameters, "\n")

                            # recreate modified MPO
                            finiteMPO = mpoHandle(QFTModel)
                        end
                    end
                end

                # ------------------------------------------------------------

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U * S, ((1, 2), (3,)))
                V = permute(V, ((1, 2), (3,)))

                # compute error
                v = @tensor theta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1],
                                                      finiteMPO[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1])
            end

            # store bogParameters
            storeBogoliubovParameters = vcat(storeBogoliubovParameters,
                                             reshape(bogParameters, 1,
                                                     length(bogParameters)))

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO)
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal)
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal)

            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end])
            # energyConvergenceNew = norm(newEigsEnergies .- oldEigsEnergies);
            # display([energyConvergence energyConvergenceNew])
            alg.verbosePrint > 0 &&
                @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n",
                        loopCounter,
                        mpoExpVal,
                        energyConvergence)
            if energyConvergence < alg.convTolE
                runOptimizationDMRG = false
            end

            # increase loopCounter and update oldEigsEnergies
            loopCounter += 1
            oldEigsEnergies = newEigsEnergies
        end

        display(bogParameters)

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO)
        @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance)

        # re-randomize finiteMPS if non-eigenstate was found
        if energyVariance > 1e-0
            if optimizationLoopCounter < maxOptimSteps
                @printf("\nre-randomizing MPS...\n")
                for idxMPS in eachindex(finiteMPS)
                    finiteMPS[idxMPS] += 0.1 *
                                         randn(ComplexF64, codomain(finiteMPS[idxMPS]),
                                               domain(finiteMPS[idxMPS]))
                end
                finiteMPS = normalizeMPS(finiteMPS)
                finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                                     compressionAlg = "zipUp")
            else
                runOptimization = false
            end
        else
            runOptimization = false
        end

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1
    end
    # @printf("\n")

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end]
    return finiteMPS, finalEnergy, storeBogoliubovParameters, ϵs
end

function find_groundstate(finiteMPS::SparseMPS, finiteMPO::SparseMPO, alg::DMRG2)
    return find_groundstate!(copy(finiteMPS), finiteMPO, alg)
end

function find_groundstate(finiteMPS::SparseMPS, mpoHandle::Function,
                          QFTModel::AbstractQFTModel, alg::DMRG2BO)
    return find_groundstate!(copy(finiteMPS), mpoHandle, QFTModel, alg)
end

function find_excitedstate!(finiteMPS::SparseMPS, finiteMPO::SparseMPO,
                            previoMPS::Vector{<:SparseMPS}, alg::DMRG2)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                             compressionAlg = "zipUp")
    end

    # set penaltyWeights
    penaltyWeights = 1e2 * ones(length(previoMPS))

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0]

    # store vector of truncation errors
    ϵs = ones(Float64, length(finiteMPS) - 1)

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1
    maxOptimSteps = 1
    runOptimization = true
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO)

        # initialize projector environments
        projEnvsL, projEnvsR = initializePROEnvironments(finiteMPS, previoMPS)

        # store vector of truncation errors
        ϵs = ones(Float64, length(finiteMPS) - 1)

        # main DMRG loop
        loopCounter = 1
        runOptimizationDMRG = true
        while runOptimizationDMRG

            # sweep L ---> R
            for siteIdx in 1:+1:(length(finiteMPS) - 1)

                # construct initial theta
                thetaN = permute(finiteMPS[siteIdx] *
                                 permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                 ((1, 2),
                                  (3, 4)))

                # construct thetas for orthStates
                thetasO = Vector{TensorMap{ComplexF64}}(undef, length(previoMPS))
                for orthMPO in eachindex(previoMPS)
                    thetasO[orthMPO] = permute(previoMPS[orthMPO][siteIdx] *
                                               permute(previoMPS[orthMPO][siteIdx + 1],
                                                       ((1,), (2, 3))),
                                               ((1, 2),
                                                (3, 4)))
                end

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(thetaN,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2_X(x,
                                     siteIdx,
                                     mpoEnvL,
                                     finiteMPO,
                                     mpoEnvR,
                                     projEnvsL,
                                     thetasO,
                                     projEnvsR,
                                     penaltyWeights)
                end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1]

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U, ((1, 2), (3,)))
                V = permute(S * V, ((1, 2), (3,)))

                # compute error
                v = @tensor newTheta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx],
                                                      finiteMPS[siteIdx],
                                                      finiteMPO[siteIdx],
                                                      finiteMPS[siteIdx])

                # shift orthogonality center of previoMPS to the right
                for orthIdx in eachindex(previoMPS)
                    (Q, R) = leftorth(previoMPS[orthIdx][siteIdx], ((1, 2), (3,));
                                      alg = QRpos())
                    previoMPS[orthIdx][siteIdx + 0] = permute(Q, ((1, 2), (3,)))
                    previoMPS[orthIdx][siteIdx + 1] = permute(R *
                                                              permute(previoMPS[orthIdx][siteIdx + 1],
                                                                      ((1,), (2, 3))),
                                                              ((1, 2),
                                                               (3,)))
                end

                # update projEnvsL
                for orthIdx in eachindex(previoMPS)
                    projEnvsL[orthIdx][siteIdx + 1] = update_MPSEnvL(projEnvsL[orthIdx][siteIdx],
                                                                     finiteMPS[siteIdx],
                                                                     previoMPS[orthIdx][siteIdx])
                end
            end

            # sweep L <--- R
            for siteIdx in (length(finiteMPS) - 1):-1:1

                # construct initial theta
                thetaN = permute(finiteMPS[siteIdx] *
                                 permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                 ((1, 2),
                                  (3, 4)))

                # construct thetas for orthStates
                thetasO = Vector{TensorMap{ComplexF64}}(undef, length(previoMPS))
                for orthMPO in eachindex(previoMPS)
                    thetasO[orthMPO] = permute(previoMPS[orthMPO][siteIdx] *
                                               permute(previoMPS[orthMPO][siteIdx + 1],
                                                       ((1,), (2, 3))),
                                               ((1, 2),
                                                (3, 4)))
                end

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(thetaN,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2_X(x,
                                     siteIdx,
                                     mpoEnvL,
                                     finiteMPO,
                                     mpoEnvR,
                                     projEnvsL,
                                     thetasO,
                                     projEnvsR,
                                     penaltyWeights)
                end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1]

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U * S, ((1, 2), (3,)))
                V = permute(V, ((1, 2), (3,)))

                # compute error
                v = @tensor newTheta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1],
                                                      finiteMPO[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1])

                # shift orthogonality center of previoMPS to the right
                for orthIdx in eachindex(previoMPS)
                    (L, Q) = rightorth(previoMPS[orthIdx][siteIdx + 1], ((1,), (2, 3));
                                       alg = LQpos())
                    previoMPS[orthIdx][siteIdx + 0] = permute(permute(previoMPS[orthIdx][siteIdx + 0],
                                                                      ((1, 2), (3,))) * L,
                                                              ((1, 2),
                                                               (3,)))
                    previoMPS[orthIdx][siteIdx + 1] = permute(Q, ((1, 2), (3,)))
                end

                # update projEnvsR
                for orthIdx in eachindex(previoMPS)
                    projEnvsR[orthIdx][siteIdx + 0] = update_MPSEnvR(projEnvsR[orthIdx][siteIdx + 1],
                                                                     finiteMPS[siteIdx + 1],
                                                                     previoMPS[orthIdx][siteIdx + 1])
                end
            end

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO)
            if imag(mpoExpVal) < 1e-12
                mpoExpVal = real(mpoExpVal)
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal)

            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end])
            alg.verbosePrint > 0 &&
                @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n",
                        loopCounter,
                        mpoExpVal,
                        energyConvergence)
            if energyConvergence < alg.convTolE
                runOptimizationDMRG = false
            end

            # increase loopCounter
            loopCounter += 1
        end

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO)
        @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance)

        # re-randomize finiteMPS if non-eigenstate was found
        if energyVariance > 1e-1
            if optimizationLoopCounter < maxOptimSteps
                @printf("\nre-randomizing MPS...\n")
                for idxMPS in eachindex(finiteMPS)
                    finiteMPS[idxMPS] += 0.1 *
                                         randn(ComplexF64, codomain(finiteMPS[idxMPS]),
                                               domain(finiteMPS[idxMPS]))
                end
                finiteMPS = normalizeMPS(finiteMPS)
                finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                                     compressionAlg = "zipUp")
            else
                runOptimization = false
            end
        else
            runOptimization = false
        end

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1
    end

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end]
    return finiteMPS, finalEnergy, ϵs
end

function find_excitedstate!(finiteMPS::SparseMPS,
                            mpoHandle::Function,
                            previoMPS::Vector{<:SparseMPS},
                            QFTModel::AbstractQFTModel,
                            alg::DMRG2BO)

    # create MPO that should be simulated
    finiteMPO = mpoHandle(QFTModel)
    println(getLinkDimsMPO(finiteMPO))

    # get bogParameters
    bogParameters = QFTModel.modelParameters.truncationParameters[:bogParameters]
    println(bogParameters)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                             compressionAlg = "zipUp")
    end

    # initialize array to store Bogoliubov parameters
    storeBogoliubovParameters = zeros(Float64, 0,
                                      QFTModel.modelParameters.truncationParameters[:kMax])
    # storeBogoliubovParameters = vcat(storeBogoliubovParameters, reshape(bogParameters, 1, length(bogParameters)));

    # set penaltyWeights
    penaltyWeights = 1e2 * ones(length(previoMPS))

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0]

    # store vector of truncation errors
    ϵs = ones(Float64, length(finiteMPS) - 1)

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1
    maxOptimSteps = 1
    runOptimization = true
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO)

        # initialize projector environments
        projEnvsL, projEnvsR = initializePROEnvironments(finiteMPS, previoMPS)

        # store vector of truncation errors
        ϵs = ones(Float64, length(finiteMPS) - 1)

        # main DMRG loop
        loopCounter = 1
        runOptimizationDMRG = true
        while runOptimizationDMRG

            # sweep L ---> R
            for siteIdx in 1:+1:(length(finiteMPS) - 1)

                # construct initial theta
                thetaN = permute(finiteMPS[siteIdx] *
                                 permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                 ((1, 2),
                                  (3, 4)))

                # construct thetas for orthStates
                thetasO = Vector{TensorMap{ComplexF64}}(undef, length(previoMPS))
                for orthMPO in eachindex(previoMPS)
                    thetasO[orthMPO] = permute(previoMPS[orthMPO][siteIdx] *
                                               permute(previoMPS[orthMPO][siteIdx + 1],
                                                       ((1,), (2, 3))),
                                               ((1, 2),
                                                (3, 4)))
                end

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(thetaN,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2_X(x,
                                     siteIdx,
                                     mpoEnvL,
                                     finiteMPO,
                                     mpoEnvR,
                                     projEnvsL,
                                     thetasO,
                                     projEnvsR,
                                     penaltyWeights)
                end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1]

                # ------------------------------------------------------------
                # perform local basis optimization to reduce entanglement

                if mod(siteIdx, 2) == 0 && loopCounter >= alg.startOptimization

                    # get physVecSpaces for squeezing operator
                    PL = space(finiteMPS[siteIdx + 0], 2)
                    PR = space(finiteMPS[siteIdx + 1], 2)
                    nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

                    # set kL and kR
                    kL = -1 * Int(siteIdx / 2)
                    kR = +1 * Int(siteIdx / 2)
                    # display([kL  kR])

                    # compute cost function pre optimization
                    costFuncPre = computeRenyiEntropy(newTheta)
                    vNEntropyPre = computeEntropy(newTheta)

                    # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
                    if alg.verbosePrint == 2
                        listOfXiValues = collect(-0.4:0.02:+0.4)
                        storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
                        storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
                        for (idx, ξ) in enumerate(listOfXiValues)
                            sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
                                                                                                           newTheta))
                            storeAnalyticGradient[idx] = analyticGradientCostFunction(ξ,
                                                                                      nMax,
                                                                                      kL,
                                                                                      kR,
                                                                                      PL,
                                                                                      PR,
                                                                                      newTheta)
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
                        plot!(renyiEntropyPlot,
                              listOfXiValues,
                              storeAnalyticGradient;
                              linewidth = 2.0,
                              label = L"\partial S(\xi)/\partial \xi",)
                        display(renyiEntropyPlot)
                    end

                    # find optimimal ξ by roots of gradient
                    # optimalXi = find_zeros(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, newTheta), -0.5, +0.5);
                    optimalXi = find_zero(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR,
                                                                            PL, PR,
                                                                            newTheta),
                                          0.0)
                    alg.verbosePrint > 0 && @show optimalXi

                    # # optimize twoSiteUnitary
                    # optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, newTheta), bogParameters[kR + 1], optimAlg,
                    #     scale! = _scale!, 
                    #     add! = _add!, 
                    # );
                    # optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes;

                    if any(abs.(optimalXi) .> 1e-3)

                        # check acceptance of optimalXi
                        newCostFunction = zeros(Float64, length(optimalXi))
                        for (idx, ξ) in enumerate(optimalXi)

                            # transform newTheta with optimalS
                            optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            optimizedTheta = applyTwoModeTransformation(optimalS, newTheta)

                            # compute cost function post optimiization
                            costFuncPost = computeRenyiEntropy(optimizedTheta)
                            vNEntropyPost = computeEntropy(optimizedTheta)
                            # display([costFuncPre costFuncPost costFuncPre > costFuncPost])
                            # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
                            # println()

                            newCostFunction[idx] = costFuncPost
                        end

                        # select optimalXi corresponding to lowest costFuncPost
                        costFuncPost, minIdx = findmin(newCostFunction)
                        optimalXi = optimalXi[minIdx]
                        # @show optimalXi

                        # decompose optimizedTheta
                        if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1],
                                           optimalXi)

                            # update two site tensor
                            optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
                            newTheta = applyTwoModeTransformation(optimalS, newTheta)
                            alg.verbosePrint > 0 && println("new optimal ξ = ", optimalXi)

                            # update QFTModel with new bogParameters
                            # bogParameters[kR + 1] -= optimalXi;
                            bogParameters[kR + 1] += optimalXi
                            QFTModel = updateBogoliubovParameters(QFTModel;
                                                                  bogoliubovRot = true,
                                                                  bogParameters = bogParameters)
                            println(bogParameters, "\n")

                            # recreate modified MPO
                            finiteMPO = mpoHandle(QFTModel)
                        end
                    end
                end

                # ------------------------------------------------------------

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U, ((1, 2), (3,)))
                V = permute(S * V, ((1, 2), (3,)))

                # compute error
                v = @tensor newTheta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx],
                                                      finiteMPS[siteIdx],
                                                      finiteMPO[siteIdx],
                                                      finiteMPS[siteIdx])

                # shift orthogonality center of previoMPS to the right
                for orthIdx in eachindex(previoMPS)
                    (Q, R) = leftorth(previoMPS[orthIdx][siteIdx], ((1, 2), (3,));
                                      alg = QRpos())
                    previoMPS[orthIdx][siteIdx + 0] = permute(Q, ((1, 2), (3,)))
                    previoMPS[orthIdx][siteIdx + 1] = permute(R *
                                                              permute(previoMPS[orthIdx][siteIdx + 1],
                                                                      ((1,), (2, 3))),
                                                              ((1, 2),
                                                               (3,)))
                end

                # update projEnvsL
                for orthIdx in eachindex(previoMPS)
                    projEnvsL[orthIdx][siteIdx + 1] = update_MPSEnvL(projEnvsL[orthIdx][siteIdx],
                                                                     finiteMPS[siteIdx],
                                                                     previoMPS[orthIdx][siteIdx])
                end
            end

            # sweep L <--- R
            for siteIdx in (length(finiteMPS) - 1):-1:1

                # construct initial theta
                thetaN = permute(finiteMPS[siteIdx] *
                                 permute(finiteMPS[siteIdx + 1], ((1,), (2, 3))),
                                 ((1, 2),
                                  (3, 4)))

                # construct thetas for orthStates
                thetasO = Vector{TensorMap{ComplexF64}}(undef, length(previoMPS))
                for orthMPO in eachindex(previoMPS)
                    thetasO[orthMPO] = permute(previoMPS[orthMPO][siteIdx] *
                                               permute(previoMPS[orthMPO][siteIdx + 1],
                                                       ((1,), (2, 3))),
                                               ((1, 2),
                                                (3, 4)))
                end

                # optimize wave function to get newAC
                maxIterKrylov = loopCounter == 1 ? alg.maxIterationsInit : alg.maxIterations
                eigenVal, eigenVec = eigsolve(thetaN,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = maxIterKrylov)) do x
                    return applyH2_X(x,
                                     siteIdx,
                                     mpoEnvL,
                                     finiteMPO,
                                     mpoEnvR,
                                     projEnvsL,
                                     thetasO,
                                     projEnvsR,
                                     penaltyWeights)
                end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1]

                # ------------------------------------------------------------
                # perform local basis optimization to reduce entanglement

                if mod(siteIdx, 2) == 0 && loopCounter > alg.startOptimization

                    # get physVecSpaces for squeezing operator
                    PL = space(finiteMPS[siteIdx + 0], 2)
                    PR = space(finiteMPS[siteIdx + 1], 2)
                    nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

                    # set kL and kR
                    kL = -1 * Int(siteIdx / 2)
                    kR = +1 * Int(siteIdx / 2)
                    # display([kL  kR])

                    # compute cost function pre optimization
                    costFuncPre = computeRenyiEntropy(newTheta)
                    vNEntropyPre = computeEntropy(newTheta)

                    # see landscape of ξ and compute analytic gradient of the cost function with respect to ξ
                    if alg.verbosePrint == 2
                        listOfXiValues = collect(-0.4:0.02:+0.4)
                        storeEntanglementEntropy = zeros(Float64, length(listOfXiValues))
                        storeAnalyticGradient = zeros(Float64, length(listOfXiValues))
                        for (idx, ξ) in enumerate(listOfXiValues)
                            sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            storeEntanglementEntropy[idx] = computeRenyiEntropy(applyTwoModeTransformation(sqOp,
                                                                                                           newTheta))
                            storeAnalyticGradient[idx] = analyticGradientCostFunction(ξ,
                                                                                      nMax,
                                                                                      kL,
                                                                                      kR,
                                                                                      PL,
                                                                                      PR,
                                                                                      newTheta)
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
                        plot!(renyiEntropyPlot,
                              listOfXiValues,
                              storeAnalyticGradient;
                              linewidth = 2.0,
                              label = L"\partial S(\xi)/\partial \xi",)
                        display(renyiEntropyPlot)
                    end

                    # find optimimal ξ by roots of gradient
                    # optimalXi = find_zeros(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR, PL, PR, newTheta), -0.5, +0.5);
                    optimalXi = find_zero(ξ -> analyticGradientCostFunction(ξ, nMax, kL, kR,
                                                                            PL, PR,
                                                                            newTheta),
                                          0.0)
                    alg.verbosePrint > 0 && @show optimalXi

                    # # optimize twoSiteUnitary
                    # optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, newTheta), bogParameters[kR + 1], optimAlg,
                    #     scale! = _scale!, 
                    #     add! = _add!, 
                    # );
                    # optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes;

                    if any(abs.(optimalXi) .> 1e-3)

                        # check acceptance of optimalXi
                        newCostFunction = zeros(Float64, length(optimalXi))
                        for (idx, ξ) in enumerate(optimalXi)

                            # transform newTheta with optimalS
                            optimalS = squeezingOp(ξ, nMax, kL, kR, PL, PR)
                            optimizedTheta = applyTwoModeTransformation(optimalS, newTheta)

                            # compute cost function post optimiization
                            costFuncPost = computeRenyiEntropy(optimizedTheta)
                            vNEntropyPost = computeEntropy(optimizedTheta)
                            # display([costFuncPre costFuncPost costFuncPre > costFuncPost])
                            # display([costFuncPre costFuncPost checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1], optimalXi) ; vNEntropyPre vNEntropyPost vNEntropyPre > vNEntropyPost])
                            # println()

                            newCostFunction[idx] = costFuncPost
                        end

                        # select optimalXi corresponding to lowest costFuncPost
                        costFuncPost, minIdx = findmin(newCostFunction)
                        optimalXi = optimalXi[minIdx]
                        # @show optimalXi

                        # decompose optimizedTheta
                        if checkAcceptance(costFuncPre, costFuncPost, bogParameters[kR + 1],
                                           optimalXi)

                            # update two site tensor
                            optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR)
                            newTheta = applyTwoModeTransformation(optimalS, newTheta)
                            println("new optimal ξ = ", optimalXi)

                            # update QFTModel with new bogParameters
                            bogParameters[kR + 1] -= optimalXi
                            # bogParameters[kR + 1] += optimalXi;
                            QFTModel = updateBogoliubovParameters(QFTModel;
                                                                  bogoliubovRot = true,
                                                                  bogParameters = bogParameters)
                            alg.verbosePrint > 0 && println(bogParameters, "\n")

                            # recreate modified MPO
                            finiteMPO = mpoHandle(QFTModel)
                        end
                    end
                end

                # ------------------------------------------------------------

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  ((1, 2),
                                   (3, 4));
                                  trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr),
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = permute(U * S, ((1, 2), (3,)))
                V = permute(V, ((1, 2), (3,)))

                # compute error
                v = @tensor newTheta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U
                finiteMPS[siteIdx + 1] = V

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1],
                                                      finiteMPO[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1])

                # shift orthogonality center of previoMPS to the right
                for orthIdx in eachindex(previoMPS)
                    (L, Q) = rightorth(previoMPS[orthIdx][siteIdx + 1], ((1,), (2, 3));
                                       alg = LQpos())
                    previoMPS[orthIdx][siteIdx + 0] = permute(permute(previoMPS[orthIdx][siteIdx + 0],
                                                                      ((1, 2), (3,))) * L,
                                                              ((1, 2),
                                                               (3,)))
                    previoMPS[orthIdx][siteIdx + 1] = permute(Q, ((1, 2), (3,)))
                end

                # update projEnvsR
                for orthIdx in eachindex(previoMPS)
                    projEnvsR[orthIdx][siteIdx + 0] = update_MPSEnvR(projEnvsR[orthIdx][siteIdx + 1],
                                                                     finiteMPS[siteIdx + 1],
                                                                     previoMPS[orthIdx][siteIdx + 1])
                end
            end

            # store bogParameters
            storeBogoliubovParameters = vcat(storeBogoliubovParameters,
                                             reshape(bogParameters, 1,
                                                     length(bogParameters)))

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO)
            if imag(mpoExpVal) < 1e-12
                mpoExpVal = real(mpoExpVal)
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal)

            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end])
            alg.verbosePrint > 0 &&
                @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n",
                        loopCounter,
                        mpoExpVal,
                        energyConvergence)
            if energyConvergence < alg.convTolE
                runOptimizationDMRG = false
            end

            # increase loopCounter
            loopCounter += 1
        end

        display(bogParameters)

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO)
        @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance)

        # re-randomize finiteMPS if non-eigenstate was found
        if energyVariance > 1e-1
            if optimizationLoopCounter < maxOptimSteps
                @printf("\nre-randomizing MPS...\n")
                for idxMPS in eachindex(finiteMPS)
                    finiteMPS[idxMPS] += 0.1 *
                                         randn(ComplexF64, codomain(finiteMPS[idxMPS]),
                                               domain(finiteMPS[idxMPS]))
                end
                finiteMPS = normalizeMPS(finiteMPS)
                finiteMPS = applyMPO(finiteMPO, finiteMPS; truncErr = 1e-3,
                                     compressionAlg = "zipUp")
            else
                runOptimization = false
            end
        else
            runOptimization = false
        end

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1
    end

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end]
    return finiteMPS, finalEnergy, storeBogoliubovParameters, ϵs
end

function find_excitedstate(finiteMPS::SparseMPS, finiteMPO::SparseMPO,
                           previoMPS::Vector{SparseMPS}, alg::DMRG2)
    return find_excitedstate!(copy(finiteMPS), finiteMPO, previoMPS, alg)
end

function find_excitedstate(finiteMPS::SparseMPS,
                           mpoHandle::Function,
                           previoMPS::Vector{SparseMPS},
                           QFTModel::AbstractQFTModel,
                           alg::DMRG2BO)
    return find_excitedstate!(copy(finiteMPS), mpoHandle, previoMPS, QFTModel, alg)
end
