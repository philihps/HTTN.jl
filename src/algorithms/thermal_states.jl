
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
)

    # set number of cooling steps for imaginary time evolution
    numCoolingSteps = Int(round(β / δβ))

    # create initial thermal density matrix
    physSpaces = [space(finiteMPO[i], 2) for i in eachindex(finiteMPO)]
    thermalDensityMatrix = constructIdentiyMPO(physSpaces, ComplexSpace(1))
    thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)

    # create infinitesimal thermal density matrix
    infinitesimalTDM = initializeThermalDensityMatrix(finiteMPO, δβ = δβ, expansionOrder = expansionOrder, truncErr = truncErr, maxDim = maxDim)

    # run TDVP to evolve the pre-quench ground state with the post-quench Hamiltonian
    for timeStep in 0 : numCoolingSteps

        @printf("timeStep = %d/%d\n", timeStep, numCoolingSteps)

        # perform cooling step
        if timeStep > 0
            thermalDensityMatrix = multiplyMPOs(thermalDensityMatrix, infinitesimalTDM, truncErr = 1e-6, maxDim = 128)
            thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)
        end

    end

    return thermalDensityMatrix

end

function costFunctionGGE(
    β::Float64,
    targetEnergy::Float64,
    finiteMPO::SparseMPO;
    δβ::Float64 = 1e-2,
    expansionOrder::Int64 = 3,
    truncErr::Float64 = 1e-6,
    maxDim::Int64 = 2500,
)

    # compute thermal density matrix at inverse temperature β
    thermalDensityMatrix = produceThermalState(finiteMPO,
        β;
        δβ = δβ,
        expansionOrder = expansionOrder,
        truncErr = truncErr,
        maxDim = maxDim)

    # compute energy of thermal state
    thermalStateEnergy = expectation_values_density_matrix(thermalDensityMatrix, finiteMPO)

    # compute cost function
    costFunction = targetEnergy - thermalStateEnergy
    return costFunction

end