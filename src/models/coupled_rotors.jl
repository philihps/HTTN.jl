"""

implementation of the coupled rotors model with nearest-neighbor cosine interaction
each rotor is defined by a momentum mode with possible eigenvalues k = -N, ..., +N

"""

struct CoupledRotorsParameters <: AbstractQFTModelParameters
    truncationParameters::NamedTuple
    hamiltonianParameters::NamedTuple

    function CoupledRotorsParameters(truncationParameters::NamedTuple,
                                     hamiltonianParameters::NamedTuple)
        return new(truncationParameters, hamiltonianParameters)
    end
end

struct CoupledRotorsModel <: AbstractQFTModel
    modelParameters::CoupledRotorsParameters
    modeOccupations::Matrix{Int64}
    physSpaces::Vector{<:Union{ElementarySpace,CompositeSpace{ElementarySpace}}}
end

function CoupledRotorsModel(truncationParameters::NamedTuple,
                            hamiltonianParameters::NamedTuple)

    # construct struct to store all model parameters
    modelParameters = CoupledRotorsParameters(truncationParameters, hamiltonianParameters)

    # get hamiltonianParameters
    β = hamiltonianParameters[:β]
    ω = hamiltonianParameters[:ω]
    κ = hamiltonianParameters[:κ]

    # construct modeOccupations and physSpaces
    modeOccupations, physSpaces = constructPhysSpaces(modelParameters)

    # construct model
    return CoupledRotorsModel(modelParameters, modeOccupations, physSpaces)
end

function generate_H0(Model::CoupledRotorsModel)
    mpo_H0 = generate_H0(Model.modelParameters, Model.modeOccupations, Model.physSpaces)
    return mpo_H0
end

function generate_H1(Model::CoupledRotorsModel)
    mpo_H1 = generate_H1(Model.modelParameters, Model.modeOccupations, Model.physSpaces)
    return mpo_H1
end

function generate_MPO_cR(Model::CoupledRotorsModel)

    # construct non-interacting and interacting MPO
    mpo_H0 = generate_H0(Model)
    mpo_H1 = generate_H1(Model)

    # apply Hamiltonian prefactors and add mpo_H0 and mpo_H1
    mpo_cR = mpo_H1 == 0 ? mpo_H0 : mpo_H0 + mpo_H1

    # check if Hamiltonian is real
    realMPO = [norm(imag(mpo_cR[siteIdx])) for siteIdx in eachindex(mpo_cR)]
    if all(realMPO .< 1e-12)
        return real(mpo_cR)
    else
        return mpo_cR
    end
end

function modelSetup(modelParameters::CoupledRotorsParameters)

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    M = truncationParameters[:M]
    N = truncationParameters[:N]

    # set momentum values
    momentumModes = convert.(Int64, 1:M)
    numSites = length(momentumModes)

    # set occupation of physical sites
    occPerSite = zeros(Int64, numSites)
    for (siteIdx, momentumVal) in enumerate(momentumModes)
        occPerSite[siteIdx] = N
    end

    # combine momentum modes and maximal mode occupation numbers
    modeOccupations = vcat(reshape(momentumModes, 1, numSites),
                           reshape(occPerSite, 1, numSites))
    return modeOccupations
end

function constructPhysSpaces(modelParameters::CoupledRotorsParameters)
    """ Constructs vector spaces for physical bond indices of the MPS """

    # construct modeOccupations
    modeOccupations = modelSetup(modelParameters)
    momentumModes = modeOccupations[1, :]
    occPerSite = modeOccupations[2, :]

    # get number of momentum modes
    numSites = length(momentumModes)

    # determine physSpaces
    physSpaces = Vector{ComplexSpace}(undef, numSites)
    for siteIdx in 1:numSites
        momentumVal = momentumModes[siteIdx]
        maxOccupation = occPerSite[siteIdx]
        physSpaces[siteIdx] = ComplexSpace(2 * maxOccupation + 1)
    end
    return modeOccupations, physSpaces
end

function initializeVacuumMPS(Model::CoupledRotorsModel)
    """ construct vacuum MPS tensor with physSpaces and virtSpaces """

    # get physSpaces
    physSpaces = Model.physSpaces
    numSites = length(physSpaces)

    # set virtSpaces
    virtSpaces = fill(ComplexSpace(1), numSites + 1)

    # initialize vacuum MPS
    mpsTensors = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for siteIdx in 1:numSites
        mpsTensor = zeros(ComplexF64,
                          dim(virtSpaces[siteIdx]),
                          dim(physSpaces[siteIdx]),
                          dim(virtSpaces[siteIdx + 1]))
        mpsTensor[1, 1, 1] = 1.0
        mpsTensors[siteIdx] = TensorMap(mpsTensor,
                                        virtSpaces[siteIdx] ⊗
                                        physSpaces[siteIdx],
                                        virtSpaces[siteIdx + 1])
    end
    return SparseMPS(mpsTensors; normalizeMPS = true)
end

function initializeMPS(Model::CoupledRotorsModel, initMPS::SparseMPS)
    """ construct random MPS tensor with physVecSpaces and virtVecSpaces """

    # construct physical and virtual vector spaces for the MPS
    physSpaces = Model.physSpaces
    boundarySpaceL = oneunit(physSpaces[1])
    boundarySpaceR = oneunit(physSpaces[end])
    virtSpaces = constructVirtSpaces(Model.physSpaces, boundarySpaceL,
                                     boundarySpaceR;
                                     removeDegeneracy = true)

    numSites = length(physSpaces)
    mpsTensors = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for siteIdx in 1:numSites
        initTensor = convert(Array, initMPS[siteIdx])
        siteTensor = randn(ComplexF64,
                           virtSpaces[siteIdx] ⊗ physSpaces[siteIdx],
                           virtSpaces[siteIdx + 1])
        siteTensor = 1e-1 * convert(Array, siteTensor)
        minTensorDim = min(size(initTensor), size(siteTensor))
        siteTensor[1:minTensorDim[1], 1:minTensorDim[2], 1:minTensorDim[3]] += initTensor[1:minTensorDim[1],
                                                                                          1:minTensorDim[2],
                                                                                          1:minTensorDim[3]]
        mpsTensors[siteIdx] = TensorMap(siteTensor,
                                        virtSpaces[siteIdx] ⊗
                                        physSpaces[siteIdx],
                                        virtSpaces[siteIdx + 1])
    end
    return SparseMPS(mpsTensors; normalizeMPS = true)
end

function generateMomentumOperator(N::Int64; returnTM::Bool = false)
    """ Construct P operators acting on (2 * N + 1)-dimensional Hilbert space """

    opP = LinearAlgebra.diagm(collect((-N):(+N)))
    if returnTM == true
        opP = TensorMap(opP, ComplexSpace(2 * N + 1), ComplexSpace(2 * N + 1))
    end
    return opP
end

function generate_H0(modelParameters::CoupledRotorsParameters,
                     modeOccupations::Matrix{Int64},
                     physSpaces::Vector{<:Union{ElementarySpace,
                                                CompositeSpace{ElementarySpace}}})
    """
    Compute ∑_{i = 1}^{M} 1/2 P_i^2
    """

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters
    β = hamiltonianParameters[:β]
    ω = hamiltonianParameters[:ω]
    κ = hamiltonianParameters[:κ]

    # get momentumModes
    momentumModes = modeOccupations[1, :]
    modeOccupations = modeOccupations[2, :]
    numSites = length(physSpaces)

    # construct H0
    mpo_H0 = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for (siteIdx, momentumVal) in enumerate(momentumModes)

        # get physical vector space
        physVecSpace = physSpaces[siteIdx]
        dimHS = dim(physVecSpace)
        modeOccupation = modeOccupations[siteIdx]

        if siteIdx == 1
            mpoBlock = zeros(ComplexF64, 1, dimHS, 2, dimHS)
            mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
            mpoBlock[1, :, 2, :] = 1 / 2 * generateMomentumOperator(modeOccupation)^2
        elseif siteIdx == numSites
            mpoBlock = zeros(ComplexF64, 2, dimHS, 1, dimHS)
            mpoBlock[1, :, 1, :] = 1 / 2 * generateMomentumOperator(modeOccupation)^2
            mpoBlock[2, :, 1, :] = getIdentityOperator(dimHS)
        else
            mpoBlock = zeros(ComplexF64, 2, dimHS, 2, dimHS)
            mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
            mpoBlock[1, :, 2, :] = 1 / 2 * generateMomentumOperator(modeOccupation)^2
            mpoBlock[2, :, 2, :] = getIdentityOperator(dimHS)
        end

        # convert mpoBlock to TensorMap
        mpo_H0[siteIdx] = TensorMap(mpoBlock,
                                    ComplexSpace(size(mpoBlock, 1)) ⊗ physSpaces[siteIdx],
                                    ComplexSpace(size(mpoBlock, 3)) ⊗ physSpaces[siteIdx])
    end
    return SparseMPO(mpo_H0)
end

function generate_H1(modelParameters::CoupledRotorsParameters,
                     modeOccupations::Matrix{Int64},
                     physSpaces::Vector{<:Union{ElementarySpace,
                                                CompositeSpace{ElementarySpace}}})
    """ Construct the cosine self-interaction and nearest-neighbor cosine interaction """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    M = truncationParameters[:M]
    N = truncationParameters[:N]

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters
    β = hamiltonianParameters[:β]
    ω = hamiltonianParameters[:ω]
    κ = hamiltonianParameters[:κ]
    BC = hamiltonianParameters[:BC]

    # get momentumModes
    momentumModes = modeOccupations[1, :]
    modeOccupations = modeOccupations[2, :]

    # get number of momentum modes
    N = length(momentumModes)

    # initialize vector of individual MPOs
    storeIndividualMPOs = Vector{SparseMPO}[]

    # collect all constant contributions in identity MPO
    constantFactor = ω^2 * M + κ * (M - 1)
    if BC == "DBC"
        constantFactor += 2 * κ
    elseif BC == "PBC"
        constantFactor += 1 * κ
    end
    identityMPO = constantFactor * constructIdentityMPO(physSpaces, oneunit(ComplexSpace))
    storeIndividualMPOs = vcat(storeIndividualMPOs, [identityMPO])

    # cosine self-interaction
    if ω != 0

        # add cosine self-interaction MPOs
        for siteIdx in eachindex(physSpaces)
            localOperators = localIdentityOp.(physSpaces)
            localOperators[siteIdx] = localNegShiftOperator(physSpaces[siteIdx]) +
                                      localPosShiftOperator(physSpaces[siteIdx])
            mpoShiftOperator = -0.5 * ω^2 *
                               convertLocalOperatorsToMPO(localOperators;
                                                          qnL = oneunit(ComplexSpace),
                                                          qnR = oneunit(ComplexSpace))
            storeIndividualMPOs = vcat(storeIndividualMPOs, [mpoShiftOperator])
        end
    end

    # nearest-neighbor cosine interaction
    if κ != 0

        # add nearest-neighbor cosine interaction MPOs
        for siteIdx in eachindex(physSpaces[1:(end - 1)])
            localOperators = localIdentityOp.(physSpaces)
            localOperators[siteIdx + 0] = localNegShiftOperator(physSpaces[siteIdx + 0])
            localOperators[siteIdx + 1] = localPosShiftOperator(physSpaces[siteIdx + 1])
            mpoShiftOperator = -0.5 * κ *
                               convertLocalOperatorsToMPO(localOperators;
                                                          qnL = oneunit(ComplexSpace),
                                                          qnR = oneunit(ComplexSpace))
            storeIndividualMPOs = vcat(storeIndividualMPOs, [mpoShiftOperator])

            localOperators = localIdentityOp.(physSpaces)
            localOperators[siteIdx + 0] = localPosShiftOperator(physSpaces[siteIdx + 0])
            localOperators[siteIdx + 1] = localNegShiftOperator(physSpaces[siteIdx + 1])
            mpoShiftOperator = -0.5 * κ *
                               convertLocalOperatorsToMPO(localOperators;
                                                          qnL = oneunit(ComplexSpace),
                                                          qnR = oneunit(ComplexSpace))
            storeIndividualMPOs = vcat(storeIndividualMPOs, [mpoShiftOperator])
        end

        # add term for DBC
        if BC == "DBC"

            localOperators = localIdentityOp.(physSpaces)
            localOperators[1] = localNegShiftOperator(physSpaces[1]) +
                                localPosShiftOperator(physSpaces[1])
            mpoShiftOperator = -0.5 * κ *
                               convertLocalOperatorsToMPO(localOperators;
                                                          qnL = oneunit(ComplexSpace),
                                                          qnR = oneunit(ComplexSpace))
            storeIndividualMPOs = vcat(storeIndividualMPOs, [mpoShiftOperator])

            localOperators = localIdentityOp.(physSpaces)
            localOperators[N] = localNegShiftOperator(physSpaces[N]) +
                                localPosShiftOperator(physSpaces[N])
            mpoShiftOperator = -0.5 * κ *
                               convertLocalOperatorsToMPO(localOperators;
                                                          qnL = oneunit(ComplexSpace),
                                                          qnR = oneunit(ComplexSpace))
            storeIndividualMPOs = vcat(storeIndividualMPOs, [mpoShiftOperator])
        end

        # add term for PBC
        if BC == "PBC"

            localOperators = localIdentityOp.(physSpaces)
            localOperators[1] = localNegShiftOperator(physSpaces[1])
            localOperators[N] = localPosShiftOperator(physSpaces[N])
            mpoShiftOperator = -0.5 * κ *
                               convertLocalOperatorsToMPO(localOperators;
                                                          qnL = oneunit(ComplexSpace),
                                                          qnR = oneunit(ComplexSpace))
            storeIndividualMPOs = vcat(storeIndividualMPOs, [mpoShiftOperator])

            localOperators = localIdentityOp.(physSpaces)
            localOperators[1] = localPosShiftOperator(physSpaces[1])
            localOperators[N] = localNegShiftOperator(physSpaces[N])
            mpoShiftOperator = -0.5 * κ *
                               convertLocalOperatorsToMPO(localOperators;
                                                          qnL = oneunit(ComplexSpace),
                                                          qnR = oneunit(ComplexSpace))
            storeIndividualMPOs = vcat(storeIndividualMPOs, [mpoShiftOperator])
        end
    end

    # add together all MPOs
    if isempty(storeIndividualMPOs)
        mpo_H1 = 0
    else
        mpo_H1 = sum(storeIndividualMPOs)
    end
    return mpo_H1
end
