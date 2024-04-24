""" two-site DMRG algorithm """

@kwdef struct DMRG2
    bondDim::Int64 = 10
    truncErr::Float64 = 1e-6
    convTolE::Float64 = 1e-6
    eigsTol::Float64 = 1e-16
    maxIterations::Int64 = 1
    subspaceExpansion::Bool = true
    verbosePrint = false
end

@kwdef struct DMRG2BO
    bondDim::Int64 = 10
    truncErr::Float64 = 1e-6
    convTolE::Float64 = 1e-6
    eigsTol::Float64 = 1e-16
    maxIterations::Int64 = 1
    subspaceExpansion::Bool = true
    verbosePrint = false
end

function applyH2(X, EL, mpo1, mpo2, ER)
    @tensor X[-1 -2; -3 -4] := EL[-1, 2, 1] * X[1, 3, 5, 6] * mpo1[2, -2, 4, 3] * mpo2[4, -3, 7, 5] * ER[6, 7, -4];
    return X
end

function applyH2_X(X, siteIdx, EL, MPO, ER, PLs, Y, PRs, penaltyWeights::Vector{Float64})

    # compute H|ψexcited⟩
    @tensor X1[-1 -2; -3 -4] := EL[siteIdx][-1, 2, 1] * X[1, 3, 5, 6] * MPO[siteIdx][2, -2, 4, 3] * MPO[siteIdx + 1][4, -3, 7, 5] * ER[siteIdx + 1][6, 7, -4];

    # compute overlaps ⟨ψ0|ψexcited⟩, ⟨ψ1|ψexcited⟩, ...
    waveFunctionOverlaps = Vector{Union{Float64, ComplexF64}}(undef, length(Y));
    for orthIdx = eachindex(Y)
        waveFunctionOverlap = @tensor PLs[orthIdx][siteIdx][1, 2] * X[2, 3, 4, 5] * conj(Y[orthIdx][1, 3, 4, 6]) * PRs[orthIdx][siteIdx + 1][5, 6];
        waveFunctionOverlaps[orthIdx] = waveFunctionOverlap;
    end

    # compute ω0 P0|ψexcited⟩, ω1 P1|ψexcited⟩, ... 
    projOverlaps = Vector{TensorMap}(undef, length(Y));
    for orthIdx = eachindex(Y)
        @tensor XO[-1 -2; -3 -4] := conj(PLs[orthIdx][siteIdx][1, -1]) * Y[orthIdx][1, -2, -3, 4] * conj(PRs[orthIdx][siteIdx + 1][-4, 4]);
        projOverlaps[orthIdx] = penaltyWeights[orthIdx] * waveFunctionOverlaps[orthIdx] * XO;
    end
    
    # return sum of terms
    return X1 + sum(projOverlaps);

end

function find_groundstate!(finiteMPS::SparseMPS, finiteMPO::SparseMPO, alg::DMRG2)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        maxDimMPS = 2 * maxLinkDimsMPS(finiteMPS);
        finiteMPS = applyMPO(finiteMPO, finiteMPS, maxDim = maxDimMPS, truncErr = 1e-6, compressionAlg = "zipUp");
    end

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0];

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1;
    maxOptimSteps = 10;
    runOptimization = true;
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO);

        # main DMRG loop
        loopCounter = 1;
        runOptimizationDMRG = true;
        while runOptimizationDMRG

            # sweep L ---> R
            for siteIdx = 1 : +1 : (length(finiteMPS) - 1)

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(theta, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                        applyH2(x, mpoEnvL[siteIdx], finiteMPO[siteIdx], finiteMPO[siteIdx + 1], mpoEnvR[siteIdx + 1])
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr), alg = TensorKit.SVD());
                S /= norm(S);
                U = permute(U, (1, 2), (3, ));
                V = permute(S * V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

            end

            # sweep L <--- R
            for siteIdx = (length(finiteMPS) - 1) : -1 : 1

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(theta, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                        applyH2(x, mpoEnvL[siteIdx], finiteMPO[siteIdx], finiteMPO[siteIdx + 1], mpoEnvR[siteIdx + 1])
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr), alg = TensorKit.SVD());
                S /= norm(S);
                U = permute(U * S, (1, 2), (3, ));
                V = permute(V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1], finiteMPS[siteIdx + 1], finiteMPO[siteIdx + 1], finiteMPS[siteIdx + 1]);
                
            end

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO);
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal);
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal);
            
            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
            alg.verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, mpoExpVal, energyConvergence);
            if energyConvergence < alg.convTolE
                runOptimizationDMRG = false;
            end

            # increase loopCounter
            loopCounter += 1;

        end

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO);
        @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance);

        # re-randomize finiteMPS if non-eigenstate was found
        if energyVariance > 5e-2
            if optimizationLoopCounter < maxOptimSteps
                @printf("\nre-randomizing MPS...\n")
                for idxMPS = eachindex(finiteMPS)
                    finiteMPS[idxMPS] += 0.1 * TensorMap(randn, codomain(finiteMPS[idxMPS]), domain(finiteMPS[idxMPS]));
                end
                finiteMPS = normalizeMPS(finiteMPS);
                maxDimMPS = floor(Int, 1.3 * maxLinkDimsMPS(finiteMPS));
                finiteMPS = applyMPO(finiteMPO, finiteMPS, maxDim = maxDimMPS, truncErr = 1e-6, compressionAlg = "zipUp");
            else
                runOptimization = false;
            end
        else
            runOptimization = false;
        end

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1;

    end
    # @printf("\n")

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end];
    return finiteMPS, finalEnergy

end

# LineSearch settings
optimAlg = LBFGS(12, verbosity = 2, maxiter  = 15);

function find_groundstate!(finiteMPS::SparseMPS, mpoHandle::Function, QFTModel::AbstractQFTModel, alg::DMRG2BO)

    # create MPO that should be simulated
    finiteMPO = mpoHandle(QFTModel);

    # get bogParameters
    bogParameters = QFTModel.modelParameters.truncationParameters[:bogParameters];
    println(bogParameters)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        maxDimMPS = 2 * maxLinkDimsMPS(finiteMPS);
        finiteMPS = applyMPO(finiteMPO, finiteMPS, maxDim = maxDimMPS, truncErr = 1e-6, compressionAlg = "zipUp");
    end

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0];

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1;
    maxOptimSteps = 10;
    runOptimization = true;
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO);

        # main DMRG loop
        loopCounter = 1;
        runOptimizationDMRG = true;
        while runOptimizationDMRG

            # sweep L ---> R
            for siteIdx = 1 : +1 : (length(finiteMPS) - 1)

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(theta, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                        applyH2(x, mpoEnvL[siteIdx], finiteMPO[siteIdx], finiteMPO[siteIdx + 1], mpoEnvR[siteIdx + 1])
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                # ------------------------------------------------------------
                # perform local basis optimization to reduce entanglement

                if mod(siteIdx, 2) == 0
            
                    # get physVecSpaces for squeezing operator
                    PL = space(finiteMPS[siteIdx + 0], 2);
                    PR = space(finiteMPS[siteIdx + 1], 2);
                    nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1));

                    # set kL and kR
                    kL = -1 * Int(siteIdx / 2);
                    kR = +1 * Int(siteIdx / 2);
                    display([kL  kR])

                    # compute cost function pre optimization
                    costFuncPre = computeRenyiEntropy(newTheta);

                    # optimize twoSiteUnitary
                    optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, newTheta), bogParameters[kR], optimAlg,
                        scale! = _scale!, 
                        add! = _add!, 
                    );
                    optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes;

                    # transform newTheta with optimalS
                    optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR);
                    optimizedTheta = applyTwoModeTransformation(optimalS, newTheta);

                    # compute cost function post optimiization
                    costFuncPost = computeRenyiEntropy(optimizedTheta);
                    display([costFuncPre costFuncPost costFuncPre > costFuncPost])

                    # # decompose optimizedTheta
                    # if costFuncPre > costFuncPost
                    #     newTheta = copy(optimizedTheta);
                    # end

                    # update bogParameters
                    bogParameters[kR] = optimalXi;
                    
                end

                # ------------------------------------------------------------

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr), alg = TensorKit.SVD());
                S /= norm(S);
                U = permute(U, (1, 2), (3, ));
                V = permute(S * V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

            end

            # sweep L <--- R
            for siteIdx = (length(finiteMPS) - 1) : -1 : 1

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(theta, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                        applyH2(x, mpoEnvL[siteIdx], finiteMPO[siteIdx], finiteMPO[siteIdx + 1], mpoEnvR[siteIdx + 1])
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                # ------------------------------------------------------------
                # perform local basis optimization to reduce entanglement

                if mod(siteIdx, 2) == 0
            
                    # get physVecSpaces for squeezing operator
                    PL = space(finiteMPS[siteIdx + 0], 2);
                    PR = space(finiteMPS[siteIdx + 1], 2);
                    nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1));

                    # set kL and kR
                    kL = -1 * Int(siteIdx / 2);
                    kR = +1 * Int(siteIdx / 2);
                    display([kL  kR])

                    # compute cost function pre optimization
                    costFuncPre = computeRenyiEntropy(newTheta);

                    # optimize twoSiteUnitary
                    optimRes = optimize(x -> value_and_gradient(x, nMax, kL, kR, PL, PR, newTheta), bogParameters[kR], optimAlg,
                        scale! = _scale!, 
                        add! = _add!, 
                    );
                    optimalXi, optimCostFunc, normGrad, normGradHistory = optimRes;

                    # transform newTheta with optimalS
                    optimalS = squeezingOp(optimalXi, nMax, kL, kR, PL, PR);
                    optimizedTheta = applyTwoModeTransformation(optimalS, newTheta);

                    # compute cost function post optimiization
                    costFuncPost = computeRenyiEntropy(optimizedTheta);
                    display([costFuncPre costFuncPost costFuncPre > costFuncPost])

                    # # decompose optimizedTheta
                    # if costFuncPre > costFuncPost
                    #     newTheta = copy(optimizedTheta);
                    # end

                    # update bogParameters
                    bogParameters[kR] = optimalXi;
                    
                end

                # ------------------------------------------------------------

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr), alg = TensorKit.SVD());
                S /= norm(S);
                U = permute(U * S, (1, 2), (3, ));
                V = permute(V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1], finiteMPS[siteIdx + 1], finiteMPO[siteIdx + 1], finiteMPS[siteIdx + 1]);
                
            end

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO);
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal);
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal);
            
            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
            alg.verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, mpoExpVal, energyConvergence);
            if energyConvergence < alg.convTolE
                runOptimizationDMRG = false;
            end

            # increase loopCounter
            loopCounter += 1;

        end

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO);
        @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance);

        # re-randomize finiteMPS if non-eigenstate was found
        if energyVariance > 5e-2
            if optimizationLoopCounter < maxOptimSteps
                @printf("\nre-randomizing MPS...\n")
                for idxMPS = eachindex(finiteMPS)
                    finiteMPS[idxMPS] += 0.1 * TensorMap(randn, codomain(finiteMPS[idxMPS]), domain(finiteMPS[idxMPS]));
                end
                finiteMPS = normalizeMPS(finiteMPS);
                maxDimMPS = floor(Int, 1.3 * maxLinkDimsMPS(finiteMPS));
                finiteMPS = applyMPO(finiteMPO, finiteMPS, maxDim = maxDimMPS, truncErr = 1e-6, compressionAlg = "zipUp");
            else
                runOptimization = false;
            end
        else
            runOptimization = false;
        end

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1;

    end
    # @printf("\n")

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end];
    return finiteMPS, finalEnergy

end

function find_groundstate(finiteMPS::SparseMPS, finiteMPO::SparseMPO, alg::DMRG2)
    return find_groundstate!(copy(finiteMPS), finiteMPO, alg)
end

function find_groundstate(finiteMPS::SparseMPS, mpoHandle::Function, QFTModel::AbstractQFTModel, alg::DMRG2BO)
    return find_groundstate!(copy(finiteMPS), mpoHandle, QFTModel, alg)
end

function find_excitedstate!(finiteMPS::SparseMPS, finiteMPO::SparseMPO, previoMPS::Vector{<:SparseMPS}, alg::DMRG2)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        maxDimMPS = 2 * maxLinkDimsMPS(finiteMPS);
        finiteMPS = applyMPO(finiteMPO, finiteMPS, maxDim = maxDimMPS, truncErr = 1e-6, compressionAlg = "zipUp");
    end

    # set penaltyWeights
    penaltyWeights = 1e2 * ones(length(previoMPS));

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0];

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1;
    maxOptimSteps = 10;
    runOptimization = true;
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO);

        # initialize projector environments
        projEnvsL, projEnvsR = initializePROEnvironments(finiteMPS, previoMPS);

        # main DMRG loop
        loopCounter = 1;
        runOptimizationDMRG = true;
        while runOptimizationDMRG

            # sweep L ---> R
            for siteIdx = 1 : +1 : (length(finiteMPS) - 1)

                # construct initial theta
                thetaN = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # construct thetas for orthStates
                thetasO = Vector{TensorMap}(undef, length(previoMPS))
                for orthMPO = eachindex(previoMPS)
                    thetasO[orthMPO] = permute(previoMPS[orthMPO][siteIdx] * permute(previoMPS[orthMPO][siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));
                end

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(thetaN, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                        applyH2_X(x, siteIdx, mpoEnvL, finiteMPO, mpoEnvR, projEnvsL, thetasO, projEnvsR, penaltyWeights)
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr), alg = TensorKit.SVD());
                U = permute(U, (1, 2), (3, ));
                V = permute(S * V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

                # shift orthogonality center of previoMPS to the right
                for orthIdx = eachindex(previoMPS)
                    (Q, R) = leftorth(previoMPS[orthIdx][siteIdx], (1, 2), (3, ), alg = QRpos());
                    previoMPS[orthIdx][siteIdx + 0] = permute(Q, (1, 2), (3, ));
                    previoMPS[orthIdx][siteIdx + 1] = permute(R * permute(previoMPS[orthIdx][siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, ));
                end

                # update projEnvsL
                for orthIdx = eachindex(previoMPS)
                    projEnvsL[orthIdx][siteIdx + 1] = update_MPSEnvL(projEnvsL[orthIdx][siteIdx], finiteMPS[siteIdx], previoMPS[orthIdx][siteIdx]);
                end

            end

            # sweep L <--- R
            for siteIdx = (length(finiteMPS) - 1) : -1 : 1

                # construct initial theta
                thetaN = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # construct thetas for orthStates
                thetasO = Vector{TensorMap}(undef, length(previoMPS))
                for orthMPO = eachindex(previoMPS)
                    thetasO[orthMPO] = permute(previoMPS[orthMPO][siteIdx] * permute(previoMPS[orthMPO][siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));
                end

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(thetaN, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                        applyH2_X(x, siteIdx, mpoEnvL, finiteMPO, mpoEnvR, projEnvsL, thetasO, projEnvsR, penaltyWeights)
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErr), alg = TensorKit.SVD());
                U = permute(U * S, (1, 2), (3, ));
                V = permute(V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1], finiteMPS[siteIdx + 1], finiteMPO[siteIdx + 1], finiteMPS[siteIdx + 1]);

                # shift orthogonality center of previoMPS to the right
                for orthIdx = eachindex(previoMPS)
                    (L, Q) = rightorth(previoMPS[orthIdx][siteIdx + 1], (1, ), (2, 3), alg = LQpos());
                    previoMPS[orthIdx][siteIdx + 0] = permute(permute(previoMPS[orthIdx][siteIdx + 0], (1, 2), (3, )) * L, (1, 2), (3, ));
                    previoMPS[orthIdx][siteIdx + 1] = permute(Q, (1, 2), (3, ));
                end

                # update projEnvsR
                for orthIdx = eachindex(previoMPS)
                    projEnvsR[orthIdx][siteIdx + 0] = update_MPSEnvR(projEnvsR[orthIdx][siteIdx + 1], finiteMPS[siteIdx + 1], previoMPS[orthIdx][siteIdx + 1]);
                end
                
            end

            # normalize MPS after DMRG step
            mpsNorm = tr(finiteMPS[1]' * finiteMPS[1]);
            finiteMPS[1] /= sqrt(mpsNorm);

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO);
            if imag(mpoExpVal) < 1e-12
                mpoExpVal = real(mpoExpVal);
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal);

            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
            alg.verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, mpoExpVal, energyConvergence);
            if energyConvergence < alg.convTolE
                runOptimizationDMRG = false;
            end

            # increase loopCounter
            loopCounter += 1;

        end

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO);
        @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance);

        # re-randomize finiteMPS if non-eigenstate was found
        if energyVariance > 5e-2
            if optimizationLoopCounter < maxOptimSteps
                @printf("\nre-randomizing MPS...\n")
                maxDimMPS = 2 * maxLinkDimsMPS(finiteMPS);
                for idxMPS = eachindex(finiteMPS)
                    finiteMPS[idxMPS] += 0.1 * TensorMap(randn, codomain(finiteMPS[idxMPS]), domain(finiteMPS[idxMPS]));
                end
                finiteMPS = normalizeMPS(finiteMPS);
                finiteMPS = applyMPO(finiteMPO, finiteMPS, maxDim = maxDimMPS, truncErr = 1e-6, compressionAlg = "zipUp");
            else
                runOptimization = false;
            end
        else
            runOptimization = false;
        end

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1;

    end
    # @printf("\n")

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end];
    return finiteMPS, finalEnergy;

end

function find_excitedstate(finiteMPS::SparseMPS, finiteMPO::SparseMPO, previoMPS::Vector{SparseMPS}, alg::DMRG2)
    return find_excitedstate!(copy(finiteMPS), finiteMPO, previoMPS, alg)
end