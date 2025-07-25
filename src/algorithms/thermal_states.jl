
function constructIdentiyMPO(physVecSpaces::Vector{<:ElementarySpace},
                             virtVecSpace::ElementarySpace)
    """ Constructs the identity MPO I with bond dimension dMPO == 1 """

    # get number of momentum modes
    numSites = length(physVecSpaces)

    # construct identityMPO
    identityMPO = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for (siteIdx, physVecSpace) in enumerate(physVecSpaces)
        dimPhysVecSpace = dim(physVecSpace)
        localMPO = zeros(ComplexF64, dim(virtVecSpace), dim(physVecSpace),
                         dim(virtVecSpace), dim(physVecSpace))
        localMPO[1, :, 1, :] = diagm(ones(dimPhysVecSpace))
        identityMPO[siteIdx] = TensorMap(localMPO, virtVecSpace ⊗ physVecSpace,
                                         virtVecSpace ⊗ physVecSpace)
    end
    return SparseMPO(identityMPO)
end

function initializeThermalDensityMatrix(finiteMPO::SparseMPO;
                                        δβ::Float64 = 1e-2,
                                        expansionOrder::Int64 = 3,
                                        truncErr::Float64 = 1e-8,
                                        maxDim::Int64 = 2500,)

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
            mpoPowers[idxT] = multiplyMPOs(finiteMPO, mpoPowers[idxT - 1]; maxDim = maxDim,
                                           truncErr = truncErr)
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

function produceThermalState(finiteMPO::SparseMPO,
                             targetEnergy::Float64;
                             δβ::Float64 = 1e-2,
                             expansionOrder::Int64 = 3,
                             truncErr::Float64 = 1e-6,
                             maxDim::Int64 = 2500,
                             convTol::Float64 = 1e-2,
                             verbosePrint::Int64 = 0,)

    # initialize array to store energy expectation value
    storeEnergy = zeros(Float64, 0, 2)

    # initialize time step
    timeStep = 0

    # create initial thermal density matrix
    physSpaces = [space(finiteMPO[i], 2) for i in eachindex(finiteMPO)]
    thermalDensityMatrix = constructIdentiyMPO(physSpaces, ComplexSpace(1))
    thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)

    # create infinitesimal thermal density matrix
    infinitesimalTDM = initializeThermalDensityMatrix(finiteMPO; δβ = δβ,
                                                      expansionOrder = expansionOrder,
                                                      truncErr = truncErr, maxDim = maxDim)

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
            newThermalDensityMatrix = multiplyMPOs(thermalDensityMatrix, infinitesimalTDM;
                                                   truncErr = truncErr,
                                                   maxDim = maxDim,
                                                   compressionAlg = "zipUp")

            # increase timeStep
            newTimeStep = timeStep + 1

        elseif timeStep > 0

            # apply thermal density matrix onto itself
            newThermalDensityMatrix = multiplyMPOs(thermalDensityMatrix,
                                                   thermalDensityMatrix;
                                                   truncErr = truncErr,
                                                   maxDim = maxDim,
                                                   compressionAlg = "zipUp")

            # increase timeStep
            newTimeStep = 2 * timeStep
        end
        newThermalDensityMatrix = normalizeMPO(newThermalDensityMatrix)

        # compute energy of cooled state
        newThermalStateEnergy = expectation_values_density_matrix(newThermalDensityMatrix,
                                                                  finiteMPO)

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
        thermalDensityMatrix = multiplyMPOs(thermalDensityMatrix, infinitesimalTDM;
                                            truncErr = truncErr,
                                            maxDim = maxDim,
                                            compressionAlg = "zipUp")
        thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)

        # increase timeStep
        timeStep += 1

        # compute energy of cooled state
        thermalStateEnergy = expectation_values_density_matrix(thermalDensityMatrix,
                                                               finiteMPO)
        storeEnergy = vcat(storeEnergy, [δβ * timeStep real(thermalStateEnergy)])
    end

    @printf("target energy reached after %d cooling steps\n", timeStep)

    return thermalDensityMatrix, δβ * timeStep, storeEnergy
end

function costFunctionGGE(β::Float64,
                         targetEnergy::Float64,
                         finiteMPO::SparseMPO;
                         δβ::Float64 = 1e-2,
                         expansionOrder::Int64 = 3,
                         truncErr::Float64 = 1e-6,
                         maxDim::Int64 = 2500,
                         verbosePrint::Int64 = 0,)

    # compute thermal density matrix at inverse temperature β
    thermalDensityMatrix = produceThermalState(finiteMPO,
                                               β;
                                               δβ = δβ,
                                               expansionOrder = expansionOrder,
                                               truncErr = truncErr,
                                               maxDim = maxDim,
                                               verbosePrint = verbosePrint)

    # compute energy of thermal state
    thermalStateEnergy = real(expectation_values_density_matrix(thermalDensityMatrix,
                                                                finiteMPO))

    # compute cost function
    costFunction = thermalStateEnergy - targetEnergy
    return costFunction
end

function reducedDensityMatrixHalfSystem(finiteMPO::SparseMPO)
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
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[-1, -2, -3, 1, -5, -6, -7,
                                                               3] *
                                                      conj(finiteMPO[4][1, 2, -4, 4]) *
                                                      finiteMPO[4][3, 2, -8, 4]
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[-1, -2, -3, 1, -5, -6, -7,
                                                               3] *
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
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[-1, -2, -3, 1, -6, -7,
                                                                      -8, 3] *
                                                             conj(finiteMPO[4][1, 2, -5,
                                                                               -4]) *
                                                             finiteMPO[4][3, 2, -10, -9]
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[-1, -2, -3, -4, 1, -6,
                                                                      -7, -8, -9, 3] *
                                                             conj(finiteMPO[5][1, 2, -5, 4]) *
                                                             finiteMPO[5][3, 2, -10, 4]
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[-1, -2, -3, -4, 1, -6,
                                                                      -7, -8, -9, 3] *
                                                             conj(finiteMPO[6][1, 2, -5, 4]) *
                                                             finiteMPO[6][3, 2, -10, 4]
        @tensor mpoTrace[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := mpoTrace[-1, -2, -3, -4, 1, -6,
                                                                      -7, -8, -9, 3] *
                                                             conj(finiteMPO[7][1, 2, -5, 4]) *
                                                             finiteMPO[7][3, 2, -10, 4]
        @tensor mpoTrace[-1 -2 -3 -4; -5 -6 -7 -8] := mpoTrace[-1, -2, -3, -4, 1, -5, -6,
                                                               -7, -8, 3] *
                                                      conj(finiteMPO[8][1, 2, 5, 4]) *
                                                      finiteMPO[8][3, 2, 5, 4]
    end
    return mpoTrace
end
