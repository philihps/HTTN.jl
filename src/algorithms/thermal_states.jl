function constructIdentiyMPO(
        physVecSpaces::Vector{<:ElementarySpace},
        virtVecSpace::ElementarySpace
    )
    """ Constructs the identity MPO I with bond dimension dMPO == 1 """

    # get number of momentum modes
    numSites = length(physVecSpaces)

    # construct identityMPO
    identityMPO = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for (siteIdx, physVecSpace) in enumerate(physVecSpaces)
        dimPhysVecSpace = dim(physVecSpace)
        localMPO = zeros(
            ComplexF64, dim(virtVecSpace), dim(physVecSpace),
            dim(virtVecSpace), dim(physVecSpace)
        )
        localMPO[1, :, 1, :] = diagm(ones(dimPhysVecSpace))
        identityMPO[siteIdx] = TensorMap(
            localMPO, virtVecSpace ⊗ physVecSpace,
            virtVecSpace ⊗ physVecSpace
        )
    end
    return SparseMPO(identityMPO)
end

function initializeThermalDensityMatrix(
        finiteMPO::SparseMPO;
        δβ::Float64 = 1.0e-2,
        expansionOrder::Int64 = 3,
        truncErr::Float64 = 1.0e-8,
        maxDim::Int64 = 2500,
    )

    # construct identity MPO
    physVecSpaces = [space(finiteMPO[i], 2) for i in eachindex(finiteMPO)]
    virtVecSpace = oneunit(physVecSpaces[1])
    identityMPO = constructIdentiyMPO(physVecSpaces, virtVecSpace)

    # initialize infinitesimal thermal density matrix
    expansionCoefficients = (-δβ) .^ collect(1:expansionOrder) ./
        factorial.(collect(1:expansionOrder))
    mpoPowers = Vector{SparseMPO}(undef, expansionOrder)
    for idxT in 1:expansionOrder
        if idxT == 1
            mpoPowers[idxT] = finiteMPO
        else
            mpoPowers[idxT] = multiplyMPOs(
                finiteMPO, mpoPowers[idxT - 1]; maxDim = maxDim,
                truncErr = truncErr
            )
        end
    end
    for idxT in 1:expansionOrder
        mpoPowers[idxT] *= expansionCoefficients[idxT]
    end

    # sum up Taylor series
    thermalDensityMatrix = identityMPO + sum(mpoPowers)
    thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)
    return thermalDensityMatrix
end

function produceThermalState(
        finiteMPO::SparseMPO,
        targetEnergy::Float64;
        δβ::Float64 = 1.0e-2,
        expansionOrder::Int64 = 3,
        truncErr::Float64 = 1.0e-6,
        maxDim::Int64 = 2500,
        convTol::Float64 = 1.0e-2,
        verbosePrint::Int64 = 0,
    )

    # initialize array to store energy expectation value
    storeEnergy = zeros(Float64, 0, 2)

    # initialize time step
    timeStep = 0

    # create initial thermal density matrix
    physSpaces = [space(finiteMPO[i], 2) for i in eachindex(finiteMPO)]
    thermalDensityMatrix = constructIdentiyMPO(physSpaces, ComplexSpace(1))
    thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)

    # create infinitesimal thermal density matrix
    infinitesimalTDM = initializeThermalDensityMatrix(
        finiteMPO; δβ = δβ,
        expansionOrder = expansionOrder,
        truncErr = truncErr, maxDim = maxDim
    )

    # compute energy of cooled state
    thermalStateEnergy = expectation_values_density_matrix(thermalDensityMatrix, finiteMPO)
    storeEnergy = vcat(storeEnergy, [δβ * timeStep real(thermalStateEnergy)])

    # use exponential cooling as long as thermal state energy is above target energy
    println("exponential cooling as long as thermal state energy is above target energy")
    runExponentialCooling = true
    while runExponentialCooling
        verbosePrint == 1 && @printf("timeStep = %d\n", timeStep)

        # perform cooling step
        if timeStep == 0

            # apply infinitesimal thermal density matrix
            newThermalDensityMatrix = multiplyMPOs(
                thermalDensityMatrix, infinitesimalTDM;
                truncErr = truncErr,
                maxDim = maxDim,
                compressionAlg = "zipUp"
            )

            # increase timeStep
            newTimeStep = timeStep + 1

        elseif timeStep > 0

            # apply thermal density matrix onto itself
            newThermalDensityMatrix = multiplyMPOs(
                thermalDensityMatrix,
                thermalDensityMatrix;
                truncErr = truncErr,
                maxDim = maxDim,
                compressionAlg = "zipUp"
            )

            # increase timeStep
            newTimeStep = 2 * timeStep
        end
        newThermalDensityMatrix = normalizeMPO(newThermalDensityMatrix)

        # compute energy of cooled state
        newThermalStateEnergy = expectation_values_density_matrix(
            newThermalDensityMatrix,
            finiteMPO
        )

        # check if new thermal state energy is below target energy
        if newThermalStateEnergy > targetEnergy
            thermalDensityMatrix = newThermalDensityMatrix
            thermalStateEnergy = newThermalStateEnergy
            timeStep = newTimeStep
            storeEnergy = vcat(storeEnergy, [δβ * timeStep real(thermalStateEnergy)])
            # println([thermalStateEnergy targetEnergy (thermalStateEnergy - targetEnergy)])
        else
            runExponentialCooling = false
        end
    end

    # use linear cooling until thermal state energy is close to target energy
    println("linear cooling until thermal state energy is close to target energy")
    while (thermalStateEnergy - targetEnergy) > convTol
        verbosePrint == 1 && @printf("timeStep = %d\n", timeStep)

        # apply infinitesimal thermal density matrix
        thermalDensityMatrix = multiplyMPOs(
            thermalDensityMatrix, infinitesimalTDM;
            truncErr = truncErr,
            maxDim = maxDim,
            compressionAlg = "zipUp"
        )
        thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)

        # increase timeStep
        timeStep += 1

        # compute energy of cooled state
        thermalStateEnergy = expectation_values_density_matrix(
            thermalDensityMatrix,
            finiteMPO
        )
        storeEnergy = vcat(storeEnergy, [δβ * timeStep real(thermalStateEnergy)])
    end

    @printf("target energy reached after %d cooling steps\n", timeStep)

    return thermalDensityMatrix, δβ * timeStep, storeEnergy
end

# function costFunctionGGE(β::Float64,
#                          targetEnergy::Float64,
#                          finiteMPO::SparseMPO;
#                          δβ::Float64 = 1e-2,
#                          expansionOrder::Int64 = 3,
#                          truncErr::Float64 = 1e-6,
#                          maxDim::Int64 = 2500,
#                          verbosePrint::Int64 = 0,)

#     # compute thermal density matrix at inverse temperature β
#     thermalDensityMatrix = produceThermalState(finiteMPO,
#                                                β;
#                                                δβ = δβ,
#                                                expansionOrder = expansionOrder,
#                                                truncErr = truncErr,
#                                                maxDim = maxDim,
#                                                verbosePrint = verbosePrint)

#     # compute energy of thermal state
#     thermalStateEnergy = real(expectation_values_density_matrix(thermalDensityMatrix,
#                                                                 finiteMPO))

#     # compute cost function
#     costFunction = thermalStateEnergy - targetEnergy
#     return costFunction
# end

function approximateFullTDM(TDM::SparseMPO; truncErr::Float64 = 1.0e-6, maxDim::Int64 = 2500)
    """ Approximates ρ(β) ≈ ρ(β/2)^† * ρ(β/2) with an MPO of bond dimension maxDim """

    # make copy of TDM
    compressedMPO = copy(TDM)

    # get length of MPOs
    N = length(compressedMPO)

    # construct left and right isomorphism
    isomL = isometry(
        fuse(space(TDM[1], 1), space(TDM[1], 1)),
        space(TDM[1], 1) ⊗ space(TDM[1], 1)'
    )
    isomR = isometry(
        space(TDM[N], 3)' ⊗ space(TDM[N], 3),
        fuse(space(TDM[N], 3)' ⊗ space(TDM[N], 3))
    )

    # zip-up from left to right
    for siteIdx in 1:N
        if siteIdx < N
            @tensor localTensor[-1 -2 -3; -4 -5] := isomL[-2, 1, 3] *
                compressedMPO[siteIdx][1, 2, -4, -1] *
                conj(TDM[siteIdx][3, 2, -5, -3])
            U, S, V = tsvd(
                localTensor, (1, 2, 3), (4, 5);
                trunc = truncdim(maxDim) & truncerr(truncErr),
                alg = TensorKit.SVD()
            )
            compressedMPO[siteIdx] = permute(U, (2, 3), (4, 1))
            isomL = S * V
        else
            @tensor compressedMPO[siteIdx][-1 -2; -3 -4] := isomL[-1, 1, 3] *
                compressedMPO[siteIdx][
                1, 2, 4,
                -4,
            ] *
                conj(TDM[siteIdx][3, 2, 5, -2]) *
                isomR[4, 5, -3]
        end
    end

    # orthogonalize and return MPO
    orthogonalizeMPO!(compressedMPO, 1)
    return compressedMPO
end

function reducedDensityMatrixHalfSystem_SL(finiteMPO::SparseMPO)

    # assumes that the double-layer MPO for ρ(β) = ρ(β/2)† * ρ(β/2) has already been compressed to a single MPO
    N = length(finiteMPO)
    qnL = space(finiteMPO[1], 1)
    qnR = space(finiteMPO[N], 3)
    boundaryL = TensorMap(ones, one(qnL), qnL)
    boundaryR = TensorMap(ones, qnR', one(qnR))
    indexList = Vector{Vector{Int}}(undef, N + 2)
    cOffset = N / 2 + 1
    for idxT in eachindex(indexList)
        if idxT == 1
            indexList[idxT] = [idxT]
        elseif idxT <= length(indexList) / 2
            indexList[idxT] = [idxT - 1, -(idxT - 1), idxT, -(idxT - 1) - N / 2]
        elseif length(indexList) / 2 < idxT < length(indexList)
            indexList[idxT] = [
                cOffset + 2 * (idxT - cOffset - 1) + 0,
                cOffset + 2 * (idxT - cOffset - 1) + 1,
                cOffset + 2 * (idxT - cOffset - 1) + 2,
                cOffset + 2 * (idxT - cOffset - 1) + 1,
            ]
        elseif idxT == length(indexList)
            indexList[idxT] = [cOffset + 2 * (idxT - cOffset - 1) + 0]
        end
    end
    mpoTensor = ncon(vcat(boundaryL, finiteMPO..., boundaryR), indexList)
    mpoTensor = permute(mpoTensor, Tuple(Int.(1:(N / 2))), Tuple(Int.((N / 2 + 1):(N))))
    # mpoTensor = convert(Array, mpoTensor)
    # sizeMPOTensor = size(mpoTensor)
    # mpoMatrix = reshape(mpoTensor, prod(sizeMPOTensor[1:N]), prod(sizeMPOTensor[1:N]))
    return mpoTensor
end

function reducedDensityMatrixHalfSystem_DL(finiteMPO::SparseMPO)
    N = length(finiteMPO)
    if N == 2
        mpoTrace = ones(space(finiteMPO[1], 1), space(finiteMPO[1], 1))
        @tensor mpoTrace[-1 -2; -3 -4] := mpoTrace[1, 3] *
            conj(finiteMPO[1][1, 2, -2, -1]) *
            finiteMPO[1][3, 2, -4, -3]
        @tensor mpoTrace[-1; -2] := mpoTrace[-1, 1, -2, 3] *
            conj(finiteMPO[2][1, 2, 5, 4]) *
            finiteMPO[2][3, 2, 5, 4]
    elseif N == 4
        mpoTrace = ones(space(finiteMPO[1], 1), space(finiteMPO[1], 1))
        @tensor mpoTrace[-1 -2; -3 -4] := mpoTrace[1, 3] *
            conj(finiteMPO[1][1, 2, -2, -1]) *
            finiteMPO[1][3, 2, -4, -3]
        @tensor mpoTrace[-1 -2 -3; -4 -5 -6] := mpoTrace[-1, 1, -4, 3] *
            conj(finiteMPO[2][1, 2, -3, -2]) *
            finiteMPO[2][3, 2, -6, -5]
        @tensor mpoTrace[-1 -2 -3; -4 -5 -6] := mpoTrace[-1, -2, 1, -4, -5, 3] *
            conj(finiteMPO[3][1, 2, -3, 4]) *
            finiteMPO[3][3, 2, -6, 4]
        @tensor mpoTrace[-1 -2; -3 -4] := mpoTrace[-1, -2, 1, -3, -4, 3] *
            conj(finiteMPO[4][1, 2, 5, 4]) *
            finiteMPO[4][3, 2, 5, 4]
    elseif N == 6
        mpoTrace = ones(space(finiteMPO[1], 1), space(finiteMPO[1], 1))
        @tensor mpoTrace[-1 -2; -3 -4] := mpoTrace[1, 3] *
            conj(finiteMPO[1][1, 2, -2, -1]) *
            finiteMPO[1][3, 2, -4, -3]
        @tensor mpoTrace[-1 -2 -3; -4 -5 -6] := mpoTrace[-1, 1, -4, 3] *
            conj(finiteMPO[2][1, 2, -3, -2]) *
            finiteMPO[2][3, 2, -6, -5]
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[-1, -2, 1, -5, -6, 3] *
            conj(finiteMPO[3][1, 2, -4, -3]) *
            finiteMPO[3][3, 2, -8, -7]
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[
            -1, -2, -3, 1, -5, -6, -7,
            3,
        ] *
            conj(finiteMPO[4][1, 2, -4, 4]) *
            finiteMPO[4][3, 2, -8, 4]
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[
            -1, -2, -3, 1, -5, -6, -7,
            3,
        ] *
            conj(finiteMPO[5][1, 2, -4, 4]) *
            finiteMPO[5][3, 2, -8, 4]
        @tensor mpoTrace[-1 -2 -3; -4 -5 -6] := mpoTrace[-1, -2, -3, 1, -4, -5, -6, 3] *
            conj(finiteMPO[6][1, 2, 5, 4]) *
            finiteMPO[6][3, 2, 5, 4]
    elseif N == 8
        mpoTrace = ones(space(finiteMPO[1], 1), space(finiteMPO[1], 1))
        @tensor mpoTrace[-1 -2; -3 -4] := mpoTrace[1, 3] *
            conj(finiteMPO[1][1, 2, -2, -1]) *
            finiteMPO[1][3, 2, -4, -3]
        @tensor mpoTrace[-1 -2 -3; -4 -5 -6] := mpoTrace[-1, 1, -4, 3] *
            conj(finiteMPO[2][1, 2, -3, -2]) *
            finiteMPO[2][3, 2, -6, -5]
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[-1, -2, 1, -5, -6, 3] *
            conj(finiteMPO[3][1, 2, -4, -3]) *
            finiteMPO[3][3, 2, -8, -7]
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[
            -1, -2, -3, 1, -6, -7,
            -8, 3,
        ] *
            conj(
            finiteMPO[4][
                1, 2, -5,
                -4,
            ]
        ) *
            finiteMPO[4][3, 2, -10, -9]
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[
            -1, -2, -3, -4, 1, -6,
            -7, -8, -9, 3,
        ] *
            conj(finiteMPO[5][1, 2, -5, 4]) *
            finiteMPO[5][3, 2, -10, 4]
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[
            -1, -2, -3, -4, 1, -6,
            -7, -8, -9, 3,
        ] *
            conj(finiteMPO[6][1, 2, -5, 4]) *
            finiteMPO[6][3, 2, -10, 4]
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[
            -1, -2, -3, -4, 1, -6,
            -7, -8, -9, 3,
        ] *
            conj(finiteMPO[7][1, 2, -5, 4]) *
            finiteMPO[7][3, 2, -10, 4]
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[
            -1, -2, -3, -4, 1, -5, -6,
            -7, -8, 3,
        ] *
            conj(finiteMPO[8][1, 2, 5, 4]) *
            finiteMPO[8][3, 2, 5, 4]
    end
    return mpoTrace
end
