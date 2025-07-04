
function constructIdentiyMPO(physVecSpaces::Vector{<:ElementarySpace}, virtVecSpace::ElementarySpace)
    """ Constructs the identity MPO I with bond dimension dMPO == 1 """

    # get number of momentum modes
    numSites = length(physVecSpaces)

    # construct identityMPO
    identityMPO = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for (siteIdx, physVecSpace) = enumerate(physVecSpaces)
        dimPhysVecSpace = dim(physVecSpace)
        localMPO  = zeros(ComplexF64, dim(virtVecSpace), dim(physVecSpace), dim(virtVecSpace), dim(physVecSpace))
        localMPO[1, :, 1, :] = diagm(ones(dimPhysVecSpace))
        identityMPO[siteIdx] = TensorMap(localMPO, virtVecSpace ⊗ physVecSpace, virtVecSpace ⊗ physVecSpace)
    end
    return SparseMPO(identityMPO)

end

function initializeThermalDensityMatrix(
    finiteMPO::SparseMPO;
    δβ::Float64 = 1e-2,
    expansionOrder::Int64 = 3,
    truncErr::Float64 = 1e-8,
    maxDim::Int64 = 2500,
)

    # construct identity MPO
    physVecSpaces = [space(finiteMPO[i], 2) for i in eachindex(finiteMPO)]
    virtVecSpace = oneunit(physVecSpaces[1])
    identityMPO = constructIdentiyMPO(physVecSpaces, virtVecSpace);

    # initialize infinitesimal thermal density matrix 
    expansionCoefficients = (-δβ) .^ collect(1 : expansionOrder) ./ factorial.(collect(1 : expansionOrder))
    mpoPowers = Vector{SparseMPO}(undef, expansionOrder)
    for idxT = 1 : expansionOrder
        if idxT == 1
            mpoPowers[idxT] = finiteMPO
        else
            mpoPowers[idxT] = multiplyMPOs(finiteMPO, mpoPowers[idxT - 1], maxDim = maxDim, truncErr = truncErr)
        end
    end
    for idxT = 1 : expansionOrder
        mpoPowers[idxT] *= expansionCoefficients[idxT]
    end

    # sum up Taylor series
    thermalDensityMatrix = identityMPO + sum(mpoPowers)
    thermalDensityMatrix = normalizeMPO(thermalDensityMatrix);
    return thermalDensityMatrix

end

function produceThermalState(
    finiteMPO::SparseMPO,
    β::Float64;
    δβ::Float64 = 1e-2,
    expansionOrder::Int64 = 3,
    truncErr::Float64 = 1e-6,
    maxDim::Int64 = 2500,
    verbosePrint::Int64 = 0,
)

    # set number of cooling steps for imaginary time evolution
    numCoolingSteps = Int(round(β / δβ))

    # initialize array to store energy expectation value
    storeEnergy = zeros(Float64, 0, 2)

    # create initial thermal density matrix
    physSpaces = [space(finiteMPO[i], 2) for i in eachindex(finiteMPO)]
    thermalDensityMatrix = constructIdentiyMPO(physSpaces, ComplexSpace(1))
    thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)

    # create infinitesimal thermal density matrix
    infinitesimalTDM = initializeThermalDensityMatrix(finiteMPO, δβ = δβ, expansionOrder = expansionOrder, truncErr = truncErr, maxDim = maxDim)

    # run TDVP to evolve the pre-quench ground state with the post-quench Hamiltonian
    for timeStep in 0 : numCoolingSteps

        verbosePrint == 1 && @printf("timeStep = %d/%d\n", timeStep, numCoolingSteps)

        # perform cooling step
        if timeStep > 0
            thermalDensityMatrix = multiplyMPOs(thermalDensityMatrix, infinitesimalTDM, truncErr = 1e-6, maxDim = 128)
            thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)
        end

        # compute energy of cooled state
        energyExpectationValue = expectation_values_density_matrix(thermalDensityMatrix, finiteMPO)
        storeEnergy = vcat(storeEnergy, [δβ * timeStep real(energyExpectationValue)])

    end

    return thermalDensityMatrix, storeEnergy

end

function costFunctionGGE(
    β::Float64,
    targetEnergy::Float64,
    finiteMPO::SparseMPO;
    δβ::Float64 = 1e-2,
    expansionOrder::Int64 = 3,
    truncErr::Float64 = 1e-6,
    maxDim::Int64 = 2500,
    verbosePrint::Int64 = 0,
)

    # compute thermal density matrix at inverse temperature β
    thermalDensityMatrix = produceThermalState(finiteMPO,
        β;
        δβ = δβ,
        expansionOrder = expansionOrder,
        truncErr = truncErr,
        maxDim = maxDim,
        verbosePrint = verbosePrint
    )

    # compute energy of thermal state
    thermalStateEnergy = real(expectation_values_density_matrix(thermalDensityMatrix, finiteMPO))

    # compute cost function
    costFunction = thermalStateEnergy - targetEnergy
    return costFunction

end

function reducedDensityMatrixHalfSystem(finiteMPO::SparseMPO)
    N = length(finiteMPO)
    qnL = space(finiteMPO[1], 1)
    qnR = space(finiteMPO[N], 3)
    boundaryL = TensorMap(ones, one(qnL), qnL)
    boundaryR = TensorMap(ones, qnR', one(qnR))
    indexList = Vector{Vector{Int}}(undef, N + 2)
    cOffset = N/2 + 1
    for idxT in eachindex(indexList)
        if idxT == 1
            indexList[idxT] = [idxT]
        elseif idxT <= length(indexList) / 2
            indexList[idxT] = [idxT - 1, -(idxT - 1), idxT, -(idxT - 1) - N/2]
        elseif length(indexList) / 2 < idxT < length(indexList)
            indexList[idxT] = [cOffset + 2 * (idxT - cOffset - 1) + 0, cOffset + 2 * (idxT - cOffset - 1) + 1, cOffset + 2 * (idxT - cOffset - 1) + 2, cOffset + 2 * (idxT - cOffset - 1) + 1]
        elseif idxT == length(indexList)
            indexList[idxT] = [cOffset + 2 * (idxT - cOffset - 1) + 0]
        end
    end
    mpoTensor = ncon(vcat(boundaryL, finiteMPO..., boundaryR), indexList)
    mpoTensor = permute(mpoTensor, Tuple(Int.(1:N/2)), Tuple(Int.((N/2 + 1):(N))))
    # mpoTensor = convert(Array, mpoTensor)
    # sizeMPOTensor = size(mpoTensor)
    # mpoMatrix = reshape(mpoTensor, prod(sizeMPOTensor[1:N]), prod(sizeMPOTensor[1:N]))
    return mpoTensor
end