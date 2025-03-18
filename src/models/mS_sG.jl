### moved and rewrote definitions of mode momentum and energy from utils.jl 

function modeMomentum(k::Int64, L::Union{Int64,Float64})
    """ Function to compute the momentum corresponding to mode k """

    Pk = 2 * pi / L * k
    return Pk
end

function modeEnergy(k::Int64, L::Union{Int64,Float64}, M::Union{Int64,Float64})
    """ Function to compute the energy corresponding to mode k """

    Ek = sqrt(modeMomentum(k, L)^2 + M^2)
    return Ek
end

struct MassiveSchwingerParameters <: AbstractQFTModelParameters
    truncationParameters::NamedTuple
    hamiltonianParameters::NamedTuple

    function MassiveSchwingerParameters(truncationParameters::NamedTuple,
                                        hamiltonianParameters::NamedTuple)
        return new(truncationParameters, hamiltonianParameters)
    end
end

struct MassiveSchwingerModel <: AbstractQFTModel
    modelParameters::MassiveSchwingerParameters
    modeOccupations::Matrix{Int64}
    physSpaces::Vector{<:Union{ElementarySpace,CompositeSpace{ElementarySpace}}}
    # mpo_H0::SparseMPO
    # mpo_H1::SparseMPO
    # modelMPO::SparseMPO
end

struct SineGordonParameters <: AbstractQFTModelParameters
    truncationParameters::NamedTuple
    hamiltonianParameters::NamedTuple

    function SineGordonParameters(truncationParameters::NamedTuple,
                                  hamiltonianParameters::NamedTuple)
        return new(truncationParameters, hamiltonianParameters)
    end
end

struct SineGordonModel <: AbstractQFTModel
    modelParameters::SineGordonParameters
    modeOccupations::Matrix{Int64}
    physSpaces::Vector{<:Union{ElementarySpace,CompositeSpace{ElementarySpace}}}
    # mpo_H0::SparseMPO
    # mpo_H1::SparseMPO
    # modelMPO::SparseMPO
end

function SineGordonModel(truncationParameters::NamedTuple,
                         hamiltonianParameters::NamedTuple)

    # construct struct to store all model parameters
    modelParameters = SineGordonParameters(truncationParameters,
                                           hamiltonianParameters)

    # get hamiltonianParameters
    β = hamiltonianParameters[:β]
    λ = hamiltonianParameters[:λ]
    L = hamiltonianParameters[:L]

    # construct modeOccupations and physSpaces
    modeOccupations, physSpaces = constructPhysSpaces(modelParameters)
    return SineGordonModel(modelParameters, modeOccupations, physSpaces)
end

function MassiveSchwingerModel(truncationParameters::NamedTuple,
                               hamiltonianParameters::NamedTuple)

    # construct struct to store all model parameters
    modelParameters = MassiveSchwingerParameters(truncationParameters,
                                                 hamiltonianParameters)

    # get hamiltonianParameters
    θ = hamiltonianParameters[:θ]
    m = hamiltonianParameters[:m]
    M = hamiltonianParameters[:M]
    L = hamiltonianParameters[:L]

    # construct modeOccupations and physSpaces
    modeOccupations, physSpaces = constructPhysSpaces(modelParameters)

    # # construct non-interacting MPO
    # mpo_H0 = generate_H0(modelParameters, modeOccupations, physSpaces);

    # # construct interacting MPO
    # mpo_H1 = generate_H1(modelParameters, modeOccupations, physSpaces);

    # # apply Hamiltonian prefactors and add mpo_H0 and mpo_H1
    # if m == 0
    #     mpo_mS = mpo_H0;
    # else

    #     # use infinite size prefactor for H1
    #     convFactor = - L * m * M / (4 * pi) * exp(0.5772156649);
    #     mpo_H1 *= convFactor;

    #     # sum up H0 and H1 with corresponding prefactors
    #     mpo_mS = mpo_H0 + mpo_H1;

    # end
    return MassiveSchwingerModel(modelParameters, modeOccupations, physSpaces)
end

function updateBogoliubovParameters(Model::Union{MassiveSchwingerModel,
                                                 SineGordonModel};
                                    bogoliubovRot::Bool,
                                    bogParameters::Union{Vector{Int64},
                                                         Vector{Float64},
                                                         Vector{ComplexF64}})::Union{MassiveSchwingerModel,
                                                                                     SineGordonModel}

    # update truncationParameters
    truncationParameters = Model.modelParameters.truncationParameters
    truncationParameters = (kMax = truncationParameters[:kMax],
                            nMax = truncationParameters[:nMax],
                            nMaxZM = truncationParameters[:nMaxZM],
                            truncMethod = truncationParameters[:truncMethod],
                            modeOrdering = truncationParameters[:modeOrdering],
                            bogoliubovRot = bogoliubovRot,
                            bogParameters = bogParameters)

    # get hamiltonianParameters
    hamiltonianParameters = Model.modelParameters.hamiltonianParameters

    # construct struct to store all model parameters
    if Model isa MassiveSchwingerModel
        modelParameters = MassiveSchwingerParameters(truncationParameters,
                                                     hamiltonianParameters)
        return MassiveSchwingerModel(modelParameters, Model.modeOccupations,
                                     Model.physSpaces)
    elseif Model isa SineGordonModel
        modelParameters = SineGordonParameters(truncationParameters,
                                               hamiltonianParameters)
        return SineGordonModel(modelParameters, Model.modeOccupations,
                               Model.physSpaces)
    end
end

function getMomentumModes(Model::Union{MassiveSchwingerModel,SineGordonModel})
    return Model.modeOccupations[1, :]
end

function generate_H0(Model::Union{MassiveSchwingerModel,SineGordonModel})
    mpo_H0 = generate_H0(Model.modelParameters, Model.modeOccupations,
                         Model.physSpaces)
    return mpo_H0
end

function generate_H1(Model::Union{MassiveSchwingerModel,SineGordonModel})
    modeOccupations, physSpaces = constructPhysSpaces(Model.modelParameters)
    mpo_H1 = generate_H1(Model.modelParameters, modeOccupations, physSpaces)
    return mpo_H1
end

function generate_MPO_mS(Model::MassiveSchwingerModel)

    # get hamiltonianParameters
    θ = Model.modelParameters.hamiltonianParameters[:θ]
    m = Model.modelParameters.hamiltonianParameters[:m]
    M = Model.modelParameters.hamiltonianParameters[:M]
    L = Model.modelParameters.hamiltonianParameters[:L]

    # construct MPOs
    modeOccupations, physSpaces = constructPhysSpaces(Model.modelParameters)
    mpo_H0 = generate_H0(Model.modelParameters, modeOccupations, physSpaces)
    mpo_H1 = generate_H1(Model.modelParameters, modeOccupations, physSpaces)

    # apply Hamiltonian prefactors and add mpo_H0 and mpo_H1
    if m == 0
        mpo_mS = mpo_H0
    else

        # use infinite size prefactor for H1
        convFactor = -L * m * M / (2 * pi) * exp(0.5772156649)
        mpo_mS = mpo_H0 + convFactor * mpo_H1
    end
    return mpo_mS
end

function generate_MPO_sG(sGModel::SineGordonModel)

    # get hamiltonianParameters
    β = sGModel.modelParameters.hamiltonianParameters[:β]
    λ = sGModel.modelParameters.hamiltonianParameters[:λ]
    L = sGModel.modelParameters.hamiltonianParameters[:L]
    m = 1  ## this is the soliton mass, which is our unit of energy in the sG model. 

    # construct MPOs
    modeOccupations, physSpaces = constructPhysSpaces(sGModel.modelParameters)
    mpo_H0 = generate_H0(sGModel.modelParameters, modeOccupations, physSpaces)
    mpo_H1 = generate_H1(sGModel.modelParameters, modeOccupations, physSpaces)

    # apply Hamiltonian prefactors and add mpo_H0 and mpo_H1
    if λ == 0
        mpo_sG = mpo_H0
    else
        Δ = β^2 / (8 * π) # Eq. (A46)
        κ = (2 * gamma(Δ)) / (π * gamma(1 - Δ)) *
            ((sqrt(π) * gamma(1 / (2 * (1 - Δ)))) /
             (2 * gamma(Δ / (2 * (1 - Δ)))))^(2 -
                                              2 * Δ) # Eq. (A48)
        convFactor = -λ * L * m^2 * (2 * π / m / L)^(2 * Δ) * κ
        mpo_sG = mpo_H0 + convFactor * mpo_H1
    end

    return mpo_sG
end

function modelSetup(modelParameters::Union{MassiveSchwingerParameters,
                                           SineGordonParameters})

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    kMax = truncationParameters[:kMax]
    nMax = truncationParameters[:nMax]
    nMaxZM = truncationParameters[:nMaxZM]
    truncMethod = truncationParameters[:truncMethod]
    modeOrdering = truncationParameters[:modeOrdering]
    bogoliubovRot = truncationParameters[:bogoliubovRot]

    # set momentum values
    momentumModes = convert.(Int64, collect((-kMax):1:(+kMax)))
    numSites = length(momentumModes)

    # sort momentumModes according to abs value
    if modeOrdering
        momentumModes = sort(momentumModes; by = abs)
    end

    # set maximal occupation number and cutOff for maximal momentum in the physical modes
    if truncMethod == 1 || truncMethod == 4
        nMax = 1 * kMax
        maximalMomentumPhysicalModes = 1 * kMax
    elseif truncMethod == 2
        nMax = 2 * kMax
        maximalMomentumPhysicalModes = 2 * kMax
    elseif truncMethod == 5 || truncMethod == 6
        nMax = nMax
        maximalMomentumPhysicalModes = nMax
    end

    # set occupation of physical sites
    occPerSite = zeros(Int64, numSites)
    for (siteIdx, momentumVal) in enumerate(momentumModes)
        if momentumVal == 0
            occPerSite[siteIdx] = nMaxZM
        else
            if truncMethod == 1 ||
               truncMethod == 2 ||
               truncMethod == 4 ||
               truncMethod == 5 ||
               truncMethod == 5
                occPerSite[siteIdx] = min(nMax,
                                          Int(floor(abs(maximalMomentumPhysicalModes /
                                                        momentumVal))))
            elseif truncMethod == 3
                occPerSite[siteIdx] = nMax
            end
        end
    end
    if truncMethod == 6
        occPerSite = calculateMaximalOccupations(truncMethod, kMax)
    end

    # combine momentum modes and maximal mode occupation numbers
    modeOccupations = vcat(reshape(momentumModes, 1, numSites),
                           reshape(occPerSite, 1, numSites))
    return modeOccupations
end

function constructPhysSpaces(modelParameters::Union{MassiveSchwingerParameters,
                                                    SineGordonParameters})
    ### extend input mS and sG models
    """ Constructs vector spaces for physical bond indices of the MPS """

    # construct modeOccupations
    modeOccupations = modelSetup(modelParameters)
    momentumModes = modeOccupations[1, :]
    occPerSite = modeOccupations[2, :]

    # get number of momentum modes
    numSites = length(momentumModes)

    # determine physSpaces
    physSpaces = Vector{typeof(U1Space())}(undef, numSites)
    for siteIdx in 1:numSites
        momentumVal = momentumModes[siteIdx]
        maxOccupation = occPerSite[siteIdx]
        if momentumVal == 0
            ### distinguish between mS and sG model (I would prefer it if the input was the model instead of its parameters but changing the input didn't work because the struct model calls the function constructPhysSpaces - let's think if there's a better way). 
            if modelParameters isa MassiveSchwingerParameters
                physSpaces[siteIdx] = U1Space(0 => (maxOccupation + 1))
            elseif modelParameters isa SineGordonParameters
                physSpaces[siteIdx] = U1Space(0 => (2 * maxOccupation + 1))
            end
        else
            physSpaces[siteIdx] = U1Space([momentumVal * occup => 1
                                           for occup in 0:(+maxOccupation)])
        end
    end
    return modeOccupations, physSpaces
end

function initializeVacuumMPS(Model::Union{MassiveSchwingerModel,
                                          SineGordonModel};
                             modeOrdering::Bool = true)
    """ construct vacuum MPS tensor with physSpaces and virtSpaces """

    # get physSpaces
    physSpaces = Model.physSpaces
    numSites = length(physSpaces)

    # set virtSpaces
    virtSpaces = fill(U1Space(0 => 1), numSites + 1)

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

function initializeMPS(Model::Union{MassiveSchwingerModel,SineGordonModel},
                       initMPS::SparseMPS;
                       modeOrdering::Bool = true,)
    """ construct random MPS tensor with physVecSpaces and virtVecSpaces """

    # construct physical and virtual vector spaces for the MPS
    physSpaces = Model.physSpaces
    boundarySpaceL = U1Space(0 => 1)
    boundarySpaceR = U1Space(0 => 1)
    virtSpaces = constructVirtSpaces(Model.physSpaces, boundarySpaceL,
                                     boundarySpaceR;
                                     removeDegeneracy = true)

    numSites = length(physSpaces)
    mpsTensors = Vector{TensorMap{ComplexF64}}(undef, numSites)
    if modeOrdering
        zeroSitePos = 1
    else
        zeroSitePos = Int((numSites - 1) / 2 + 1)
    end

    for siteIdx in 1:numSites
        initTensor = 1e-0 * convert(Array, initMPS[siteIdx])
        siteTensor = randn(ComplexF64,
                           virtSpaces[siteIdx] ⊗ physSpaces[siteIdx],
                           virtSpaces[siteIdx + 1])
        if siteIdx == zeroSitePos
            siteTensor = 1e-2 * convert(Array, siteTensor)
        else
            siteTensor = 1e-2 * convert(Array, siteTensor)
        end
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

function generate_H0_Part_A(modelParameters::Union{MassiveSchwingerParameters,
                                                   SineGordonParameters},
                            modeOccupations::Matrix{Int64},
                            physSpaces::Vector{<:Union{ElementarySpace,
                                                       CompositeSpace{ElementarySpace}}})
    """
    Compute ∑_{k>=0} E_k . [(μ_k^2 + ν_k^2) . A†_{-k} A_{-k})] + 𝒪
    """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    bogoliubovRot = truncationParameters[:bogoliubovRot]
    if bogoliubovRot
        bogParameters = truncationParameters[:bogParameters]
    end

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters
    # distinguish between mS and sG model 
    if modelParameters isa MassiveSchwingerParameters
        θ = hamiltonianParameters[:θ]
        m = hamiltonianParameters[:m]
        M = hamiltonianParameters[:M]
        β = sqrt(4 * pi)    # β = sqrt(4 * pi) for the mS model 
        # R = 1 / sqrt(4 * pi);    # R is the compactification radius R = 1/β which for the mS model is 1/sqrt(4 * pi)
    end
    if modelParameters isa SineGordonParameters
        θ = 0.0    # default value for sG is 0
        m = 1.0    # m is the fermion mass of the fermionic equivalent model, which for the sG model is the soliton mass with is 1
        M = 0.0    # M is the boson mass used to define the harmonic mode frequencies, which for the sG model is 0
        β = hamiltonianParameters[:β]
        # @printf("β = %f \n", β);
    end
    L = hamiltonianParameters[:L]

    # get momentumModes
    momentumModes = modeOccupations[1, :]
    numSites = length(physSpaces)

    # construct H0 Part A
    mpo_H0_Part_A = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for (siteIdx, momentumVal) in enumerate(momentumModes)

        # get physical vector space
        physVecSpace = physSpaces[siteIdx]
        dimHS = dim(physVecSpace)

        if bogoliubovRot
            kIdx = abs(momentumVal) + 1
            ξ = bogParameters[kIdx]
            μ = cosh(ξ)
            ν = sinh(ξ)
        end

        if momentumVal == 0
            # Compute zero-mode operator 𝒪

            if modelParameters isa MassiveSchwingerParameters
                modeFactor = M

                if numSites != 1
                    if bogoliubovRot
                        modeFactor *= (μ^2 + ν^2)
                    end

                    if siteIdx == 1
                        mpoBlock = zeros(ComplexF64, 1, dimHS, 2, dimHS)
                        mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
                        mpoBlock[1, :, 2, :] = modeFactor *
                                               getNumberOperator(dimHS - 1)
                    elseif siteIdx == numSites
                        mpoBlock = zeros(ComplexF64, 2, dimHS, 1, dimHS)
                        mpoBlock[1, :, 1, :] = modeFactor *
                                               getNumberOperator(dimHS - 1)
                        mpoBlock[2, :, 1, :] = getIdentityOperator(dimHS)
                    else
                        mpoBlock = zeros(ComplexF64, 2, dimHS, 2, dimHS)
                        mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
                        mpoBlock[1, :, 2, :] = modeFactor *
                                               getNumberOperator(dimHS - 1)
                        mpoBlock[2, :, 2, :] = getIdentityOperator(dimHS)
                    end
                else
                    mpoBlock = zeros(ComplexF64, 1, dimHS, 1, dimHS)
                    if bogoliubovRot
                        # 𝒪 = M [(μ² + ν²).A^†A + μν.((A^†)^2 + A^2) + ν².Id]
                        partA = (μ^2 + ν^2) * getNumberOperator(dimHS - 1)
                        partB = μ * ν *
                                (getCreationOperator(dimHS - 1) *
                                 getCreationOperator(dimHS - 1) +
                                 getAnnihilationOperator(dimHS - 1) *
                                 getAnnihilationOperator(dimHS - 1))
                        partC = ν^2 * getIdentityOperator(dimHS)
                        mpoBlock[1, :, 1, :] = modeFactor * (partA + partB + partC)
                    else
                        # 𝒪 for μ = 1 and ν = 0
                        mpoBlock[1, :, 1, :] = modeFactor *
                                               getNumberOperator(dimHS - 1)
                    end
                end
            end
            if modelParameters isa SineGordonParameters
                # Compute O_{sG} = (1 / 2) L . Π0²

                # set modeFactor
                modeFactor = 1 / (2 * L) # factor in first term in Eq. (A13)
                R = 1.0 / β

                # compute occupation per site for the zeroMode
                nMaxZM = Int(0.5 * (dimHS - 1))
                # Section 5 Appendix
                if numSites != 1
                    if siteIdx == 1
                        mpoBlock = zeros(ComplexF64, 1, dimHS, 2, dimHS)
                        mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
                        mpoBlock[1, :, 2, :] = modeFactor *
                                               generateOperatorΠ0(nMaxZM, R)^2
                    elseif siteIdx == numSites
                        mpoBlock = zeros(ComplexF64, 2, dimHS, 1, dimHS)
                        mpoBlock[1, :, 1, :] = modeFactor *
                                               generateOperatorΠ0(nMaxZM, R)^2
                        mpoBlock[2, :, 1, :] = getIdentityOperator(dimHS)
                    else
                        mpoBlock = zeros(ComplexF64, 2, dimHS, 2, dimHS)
                        mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
                        mpoBlock[1, :, 2, :] = modeFactor *
                                               generateOperatorΠ0(nMaxZM, R)^2
                        mpoBlock[2, :, 2, :] = getIdentityOperator(dimHS)
                    end
                else
                    mpoBlock = zeros(ComplexF64, 1, dimHS, 1, dimHS)
                    mpoBlock[1, :, 1, :] = modeFactor *
                                           generateOperatorΠ0(nMaxZM, R)^2
                end
            end

        else
            # Compute ∑_k E_k . [(μ_k^2 + ν_k^2) . a†_k a_k)]
            # get ordering of QNs in physVecSpace
            qnSectors = physVecSpace.dims
            phyVecSpaceOrdering = [productSector.charge
                                   for productSector in keys(qnSectors)]

            # set modeFactor
            modeFactor = modeEnergy(momentumVal, L, M) # E_k

            # get Bogoliubov rotation parameters
            if bogoliubovRot
                modeFactor *= (μ^2 + ν^2)
            end

            if siteIdx == 1
                mpoBlock = zeros(ComplexF64, 1, dimHS, 2, dimHS)
                mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
                numberOperator = diagm(abs.(phyVecSpaceOrdering ./ momentumVal))
                mpoBlock[1, :, 2, :] = modeFactor * numberOperator
            elseif siteIdx == numSites
                mpoBlock = zeros(ComplexF64, 2, dimHS, 1, dimHS)
                numberOperator = diagm(abs.(phyVecSpaceOrdering ./ momentumVal))
                mpoBlock[1, :, 1, :] = modeFactor * numberOperator
                mpoBlock[2, :, 1, :] = getIdentityOperator(dimHS)
            else
                mpoBlock = zeros(ComplexF64, 2, dimHS, 2, dimHS)
                mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS)
                numberOperator = diagm(abs.(phyVecSpaceOrdering ./ momentumVal))
                mpoBlock[1, :, 2, :] = modeFactor * numberOperator
                mpoBlock[2, :, 2, :] = getIdentityOperator(dimHS)
            end
        end

        # convert mpoBlock to TensorMap
        mpo_H0_Part_A[siteIdx] = TensorMap(mpoBlock,
                                           U1Space(0 => size(mpoBlock, 1)) ⊗
                                           physSpaces[siteIdx],
                                           U1Space(0 => size(mpoBlock, 3)) ⊗
                                           physSpaces[siteIdx])
    end
    return SparseMPO(mpo_H0_Part_A)
end

function generate_H0_Part_B(modelParameters::Union{MassiveSchwingerParameters,
                                                   SineGordonParameters},
                            modeOccupations::Matrix{Int64},
                            physSpaces::Vector{<:Union{ElementarySpace,
                                                       CompositeSpace{ElementarySpace}}})
    """
    Compute 2 . ∑_{k>0} E_k . μ_k . ν_k . [A†_{-k} A†_{+k} + A_{+k} A_{-k}]
    """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    kMax = truncationParameters[:kMax]
    bogoliubovRot = truncationParameters[:bogoliubovRot]
    if bogoliubovRot
        bogParameters = truncationParameters[:bogParameters]
    end

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters
    if modelParameters isa MassiveSchwingerParameters
        M = hamiltonianParameters[:M]
    elseif modelParameters isa SineGordonParameters
        M = 0.0
    end
    L = hamiltonianParameters[:L]

    # get momentumModes
    momentumModes = modeOccupations[1, :]
    numSites = length(physSpaces)

    # construct H0 part B
    storeIndividualMPOs = Vector{SparseMPO}(undef, kMax + 1)
    for (kIdx, kVal) in enumerate(collect(0:kMax))
        if kVal == 0
            localOperators = Vector{TensorMap{ComplexF64}}(undef, numSites)
            for (siteIdx, momentumVal) in enumerate(momentumModes)
                if momentumVal == 0
                    AnAn = mergeLocalOperators(localAnnihilationOp(momentumVal,
                                                                   physSpaces[siteIdx]),
                                               localAnnihilationOp(momentumVal,
                                                                   physSpaces[siteIdx]))
                    localOperators[siteIdx] = AnAn
                else
                    localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx])
                end
            end
            mpoAnAnZM = convertLocalOperatorsToMPO(localOperators)

            localOperators = Vector{TensorMap{ComplexF64}}(undef, numSites)
            for (siteIdx, momentumVal) in enumerate(momentumModes)
                if momentumVal == 0
                    CrCr = mergeLocalOperators(localCreationOp(momentumVal,
                                                               physSpaces[siteIdx]),
                                               localCreationOp(momentumVal,
                                                               physSpaces[siteIdx]))
                    localOperators[siteIdx] = CrCr
                else
                    localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx])
                end
            end
            mpoCrCrZM = convertLocalOperatorsToMPO(localOperators)

            # get Bogoliubov rotation parameters (this is not checked for complex ξ)
            ξ = bogParameters[abs(kVal) + 1]

            mpoAnAnZM *= M * 1 / 2 * sinh(2 * ξ)
            mpoCrCrZM *= M * 1 / 2 * sinh(2 * ξ)

            # store sum of MPOs
            storeIndividualMPOs[kIdx] = mpoAnAnZM + mpoCrCrZM

        else
            # construct momentum-preserving MPO for [A(-k).A(+k)]
            localOperators = Vector{TensorMap{ComplexF64}}(undef, numSites) #XXX: Why is k=0 included?
            for (siteIdx, momentumVal) in enumerate(momentumModes)
                if abs(momentumVal) == kVal
                    localOperators[siteIdx] = localAnnihilationOp(momentumVal,
                                                                  physSpaces[siteIdx])
                else
                    localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx])
                end
            end

            # construct momentum-preserving MPO using a kroneckerDelta MPS
            mpoAnAn = convertLocalOperatorsToMPO(localOperators)

            # construct momentum-preserving MPO for [A(-k)^†.A(+k)^†]
            localOperators = Vector{TensorMap{ComplexF64}}(undef, numSites)
            for (siteIdx, momentumVal) in enumerate(momentumModes)
                if abs(momentumVal) == kVal
                    localOperators[siteIdx] = localCreationOp(momentumVal,
                                                              physSpaces[siteIdx])
                else
                    localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx])
                end
            end

            # construct momentum-preserving MPO using a kroneckerDelta MPS
            mpoCrCr = convertLocalOperatorsToMPO(localOperators)

            # get Bogoliubov rotation parameters (this is not checked for complex ξ)
            ξ = bogParameters[abs(kVal) + 1]

            mpoAnAn *= modeEnergy(kVal, L, M) * sinh(2 * ξ)
            mpoCrCr *= modeEnergy(kVal, L, M) * sinh(2 * ξ)

            # store sum of MPOs
            storeIndividualMPOs[kIdx] = mpoAnAn + mpoCrCr
        end
    end

    mpo_H0_Part_B = sum(storeIndividualMPOs)

    return mpo_H0_Part_B
end

function generate_H0_Part_C(modelParameters::Union{MassiveSchwingerParameters,
                                                   SineGordonParameters},
                            physSpaces::Vector{<:Union{ElementarySpace,
                                                       CompositeSpace{ElementarySpace}}})
    """
    Compute 2 . ∑_{k>0} E_k . ν_k^2
    """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    kMax = truncationParameters[:kMax]
    bogoliubovRot = truncationParameters[:bogoliubovRot]
    if bogoliubovRot
        bogParameters = truncationParameters[:bogParameters]
    end

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters
    # distinguish between mS and sG model 
    if modelParameters isa MassiveSchwingerParameters
        M = hamiltonianParameters[:M]
    elseif modelParameters isa SineGordonParameters
        M = 0.0
    end
    L = hamiltonianParameters[:L]

    # get momentumModes
    numSites = length(physSpaces)

    # construct H0 Part C
    storeIndividualMPOs = Vector{SparseMPO}(undef, kMax + 1)
    for (kIdx, kVal) in enumerate(collect(0:kMax))

        # construct momentum-presering identity MPO (constant energy term after Bogoliubov rotation)
        localOperators = Vector{TensorMap{ComplexF64}}(undef, numSites)
        for siteIdx in 1:numSites
            localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx])
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpoIdId = convertLocalOperatorsToMPO(localOperators)

        # get Bogoliubov rotation parameters (this is not checked for complex ξ)
        ξ = bogParameters[abs(kVal) + 1]
        ν = sinh(ξ)
        if kVal == 0
            mpoIdId *= M * ν^2
        else
            mpoIdId *= modeEnergy(kVal, L, M) * 2 * ν^2
        end
        # store MPO
        storeIndividualMPOs[kIdx] = mpoIdId
    end

    # add together all MPOs
    mpo_H0_Part_C = sum(storeIndividualMPOs)

    return mpo_H0_Part_C
end

function generate_H0(modelParameters::Union{MassiveSchwingerParameters,
                                            SineGordonParameters},
                     modeOccupations::Matrix{Int64},
                     physSpaces::Vector{<:Union{ElementarySpace,
                                                CompositeSpace{ElementarySpace}}})
    """ Function to generate the non-interacting part H0 of the massive Schwinger Hamiltonian """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    bogoliubovRot = truncationParameters[:bogoliubovRot]
    kMax = modelParameters.truncationParameters[:kMax]

    # get diagonal part of H0 (part A)
    mpo_H0 = generate_H0_Part_A(modelParameters, modeOccupations, physSpaces)

    # get additional parts due to the Bogoliubov rotation
    if bogoliubovRot && kMax > 0
        mpo_H0 += generate_H0_Part_B(modelParameters, modeOccupations,
                                     physSpaces)
        mpo_H0 += generate_H0_Part_C(modelParameters,
                                     physSpaces)
    end

    if modelParameters isa SineGordonParameters
        # shift by the ground state energy of an infinite number of harmonic oscillators with frequencies 2 pi k / L 
        L = modelParameters.hamiltonianParameters[:L]
        mpo_H0 += -1 / 12 * 2 * pi / L *
                  constructIdentityMPO(physSpaces, Rep[U₁](0 => 1))
    end

    return mpo_H0
end

function G(nBra::Int64, nKet::Int64, w::Float64)::Float64
    """ Matrix element of the k-factor of V_{(±1, 0)}(x = 0, t = 0) for compactification radius R=sqrt(4π) in a massive mode basis of mass M """

    matElem = sqrt(factorial(big(nBra)) * factorial(big(nKet))) *
              sum([(-1)^(j + nKet - nBra) * w^(2 * j + nKet - nBra) /
                   factorial(big(j)) /
                   factorial(big(nBra - j)) / factorial(big(j + nKet - nBra))
                   for j in max(0, nBra - nKet):nBra])
    return convert(Float64, matElem)
end

### copied from sG 
function kappaFunc(p::Float64)
    kappaVal = 2 * gamma(p^2 / 2) / pi / gamma(1 - p^2 / 2) *
               (sqrt(pi) * gamma(1 / (2 - p^2)) / 2 /
                gamma(p^2 / (4 - 2 * p^2)))^(2 - p^2)
    return kappaVal
end

### copied from sG 
function generateOperatorΠ0(nMaxZM::Int64, R::Float64; returnTM::Bool = false)
    """ Construct Π0 mode operators acting on (2 * nMaxZM + 1)-dimensional Hilbert space """

    opPi0 = 1 / R * LinearAlgebra.diagm(collect((-nMaxZM):nMaxZM))
    if returnTM == true
        opPi0 = TensorMap(opPi0, ComplexSpace(2 * nMaxZM + 1),
                          ComplexSpace(2 * nMaxZM + 1))
    end
    return opPi0
end

# function F_mS(nBra::Int64, nKet::Int64, j::Int64, k::Union{Int64, Float64}, alpha::Union{Int64, Float64}, L::Float64, M::Union{Int64, Float64})::Float64
#     """ Matrix element of the k-factor of V_{(±1, 0)}(x = 0, t = 0) for compactification radius R=sqrt(4π) in a massive mode basis of mass M """

#     matEL = convert(Float64, (-1)^(j + nKet - nBra) * (alpha / sqrt(2 * modeEnergy(k, L, M) * L))^(2*j + nKet - nBra) * sqrt(factorial(big(nBra)) * factorial(big(nKet))) / factorial(big(j)) / factorial(big(nBra - j)) / factorial(big(j + nKet - nBra)));
#     return matEL
# end

### this is the new function 
function localVertexOp(modelParameters,
                       physVecSpace::Union{ElementarySpace,
                                           CompositeSpace{ElementarySpace}},
                       k::Int64, n::Int64, β::Float64, M::Float64, L::Float64)
    """ 
    Construct normal-ordered local vertex operator in
    - Fock basis: V^A(w) = G(w) or
    - Transformed basis: V^B = exp([1 - (μ + ν)^2]w^2/2) . V^A(w . (μ + ν))
    for w = n . α/√(2 ω_k L)

    :Params:
    - n: labels eigesntate of Π0
    """

    # construct kroneckerDelta space
    kronDelSpace = removeDegeneracyQN(fuse(physVecSpace,
                                           conj(flip(physVecSpace))))

    # get dimensions of physVecSpace and kronDelSpace
    dimPhysVecSpace = dim(physVecSpace)
    dimAuxVecSpace = dim(kronDelSpace)

    # compute vertex operator coefficient α
    α = convert(Float64, n * β)

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters
    bogoliubovRot = truncationParameters[:bogoliubovRot]
    w = α / sqrt(2 * modeEnergy(k, L, M) * L)
    if bogoliubovRot
        ξ = truncationParameters[:bogParameters][abs(k) + 1]
        μ = cosh(abs(ξ))
        ν = sinh(abs(ξ)) * exp(1im * angle(ξ))
        argDisplacementOp = μ * (1im * w) - ν * conj(1im * w)
        vertexOp = exp(abs(w)^2 / 2) *
                   (getDisplacementOperator(dimPhysVecSpace - 1, argDisplacementOp))
    else
        vertexOp = exp(abs(w)^2 / 2) *
                   (getDisplacementOperator(dimPhysVecSpace - 1, 1im * w))
    end

    # construct local interaction for k = 0 or k ≠ 0
    if k == 0
        if M == 0.0
            # fill interactionTensor of massless zero mode: this is a "free particle" instead of a harmonic mode, so the exponential is a jump operator between the levels 
            interactionTensor = zeros(ComplexF64, dimPhysVecSpace,
                                      dimPhysVecSpace,
                                      dimAuxVecSpace)
            if n == 0
                for rk in 1:dimPhysVecSpace
                    interactionTensor[rk, rk, 1] = 1.0
                end
            elseif n > 0
                for rk in 1:(dimPhysVecSpace - 1)
                    interactionTensor[(rk + 1), rk, 1] = 1.0
                end
            elseif n < 0
                for rk in 1:(dimPhysVecSpace - 1)
                    interactionTensor[rk, (rk + 1), 1] = 1.0
                end
            end

        elseif M != 0.0
            # fill interactionTensor for massive zero mode: this is now a harmonic mode (independently of whether it is massless or massive, there is no quantum number constraint for the zero mode, so it should be costructed differently from the nonzero modes) 
            interactionTensor = zeros(ComplexF64, dimPhysVecSpace,
                                      dimPhysVecSpace,
                                      dimAuxVecSpace)
            for nBra in 0:(dimPhysVecSpace - 1), nKet in 0:(dimPhysVecSpace - 1)
                interactionTensor[nBra + 1, nKet + 1, 1] = vertexOp[nBra + 1, nKet + 1]
            end
        end

    else

        # get ordering of QNs in physVecSpace and kronDelSpace
        physQNSectors = physVecSpace.dims
        auxQNSectors = kronDelSpace.dims
        phyVecSpaceOrdering = [productSector.charge
                               for productSector in keys(physQNSectors)]
        auxVecSpaceOrdering = [productSector.charge
                               for productSector in keys(auxQNSectors)]

        # fill interactionTensor
        interactionTensor = zeros(ComplexF64, dimPhysVecSpace, dimPhysVecSpace,
                                  dimAuxVecSpace)
        for nBra in 0:(dimPhysVecSpace - 1), nKet in 0:(dimPhysVecSpace - 1)
            braIndPos = findfirst(phyVecSpaceOrdering .== (k * nBra))
            ketIndPos = findfirst(phyVecSpaceOrdering .== (k * nKet))
            auxIndPos = findfirst(auxVecSpaceOrdering .== (k * (nBra - nKet)))
            interactionTensor[braIndPos, ketIndPos, auxIndPos] = vertexOp[braIndPos,
                                                                          ketIndPos]
        end
    end

    # convert interactionTensor to TensorMap with U1Space
    return TensorMap(interactionTensor, physVecSpace,
                     physVecSpace ⊗ kronDelSpace)
end

# function localVertexOp(modelParameters,
#                        physVecSpace::Union{ElementarySpace,
#                                            CompositeSpace{ElementarySpace}},
#                        k::Int64, n::Int64, β::Float64, M::Float64, L::Float64)
#     """ 
#     Construct normal-ordered local vertex operator in
#     - Fock basis: V^A(w) = G(w) or
#     - Transformed basis: V^B = exp([1 - (μ + ν)^2]w^2/2) . V^A(w . (μ + ν))
#     for w = n . α/√(2 ω_k L)

#     :Params:
#     - n: labels eigesntate of Π0
#     """

#     # construct kroneckerDelta space
#     kronDelSpace = removeDegeneracyQN(fuse(physVecSpace,
#                                            conj(flip(physVecSpace))))

#     # get dimensions of physVecSpace and kronDelSpace
#     dimPhysVecSpace = dim(physVecSpace)
#     dimAuxVecSpace = dim(kronDelSpace)

#     # compute vertex operator coefficient α
#     α = convert(Float64, n * β)

#     # get truncationParameters
#     truncationParameters = modelParameters.truncationParameters
#     bogoliubovRot = truncationParameters[:bogoliubovRot]
#     w = α / sqrt(2 * modeEnergy(k, L, M) * L) # parameter for the G function
#     if bogoliubovRot
#         ξ = truncationParameters[:bogParameters][abs(k) + 1]
#         μ = cosh(ξ)
#         ν = sinh(ξ)
#         factorBCH = exp(-w^2 * (ν^2 + μ * ν))
#     end

#     # construct local interaction for k = 0 or k ≠ 0
#     if k == 0
#         if M == 0.0
#             # fill interactionTensor of massless zero mode: this is a "free particle" instead of a harmonic mode, so the exponential is a jump operator between the levels 
#             interactionTensor = zeros(ComplexF64, dimPhysVecSpace,
#                                       dimPhysVecSpace,
#                                       dimAuxVecSpace)
#             if n == 0
#                 for rk in 1:dimPhysVecSpace
#                     interactionTensor[rk, rk, 1] = 1.0
#                 end
#             elseif n > 0
#                 for rk in 1:(dimPhysVecSpace - 1)
#                     interactionTensor[(rk + 1), rk, 1] = 1.0
#                 end
#             elseif n < 0
#                 for rk in 1:(dimPhysVecSpace - 1)
#                     interactionTensor[rk, (rk + 1), 1] = 1.0
#                 end
#             end

#         elseif M != 0.0
#             # fill interactionTensor for massive zero mode: this is now a harmonic mode (independently of whether it is massless or massive, there is no quantum number constraint for the zero mode, so it should be costructed differently from the nonzero modes) 
#             interactionTensor = zeros(ComplexF64, dimPhysVecSpace,
#                                       dimPhysVecSpace,
#                                       dimAuxVecSpace)
#             for nBra in 0:(dimPhysVecSpace - 1), nKet in 0:(dimPhysVecSpace - 1)
#                 if bogoliubovRot
#                     interactionTensor[nBra + 1, nKet + 1, 1] = factorBCH *
#                                                                G(nBra, nKet, w * exp(ξ))
#                 else
#                     interactionTensor[nBra + 1, nKet + 1, 1] = G(nBra, nKet, w)
#                 end
#             end
#         end

#     else

#         # get ordering of QNs in physVecSpace and kronDelSpace
#         physQNSectors = physVecSpace.dims
#         auxQNSectors = kronDelSpace.dims
#         phyVecSpaceOrdering = [productSector.charge
#                                for productSector in keys(physQNSectors)]
#         auxVecSpaceOrdering = [productSector.charge
#                                for productSector in keys(auxQNSectors)]

#         # fill interactionTensor
#         interactionTensor = zeros(ComplexF64, dimPhysVecSpace, dimPhysVecSpace,
#                                   dimAuxVecSpace)
#         for nBra in 0:(dimPhysVecSpace - 1), nKet in 0:(dimPhysVecSpace - 1)
#             braIndPos = findfirst(phyVecSpaceOrdering .== (k * nBra))
#             ketIndPos = findfirst(phyVecSpaceOrdering .== (k * nKet))
#             auxIndPos = findfirst(auxVecSpaceOrdering .== (k * (nBra - nKet)))
#             if bogoliubovRot
#                 interactionTensor[braIndPos, ketIndPos, auxIndPos] = factorBCH *
#                                                                      G(nBra,
#                                                                        nKet,
#                                                                        w *
#                                                                        exp(ξ))
#             else
#                 interactionTensor[braIndPos, ketIndPos, auxIndPos] = G(nBra,
#                                                                        nKet, w)
#             end
#         end
#     end

#     # convert interactionTensor to TensorMap with U1Space
#     return TensorMap(interactionTensor, physVecSpace,
#                      physVecSpace ⊗ kronDelSpace)
# end

function generate_H1(modelParameters::Union{MassiveSchwingerParameters,
                                            SineGordonParameters},
                     modeOccupations::Matrix{Int64},
                     physSpaces::Vector{<:Union{ElementarySpace,
                                                CompositeSpace{ElementarySpace}}})
    """ Construct the cosine interaction 
    :cos(βΦ - θ): = 1/2 . [e^{-iθ}:V_β: + e^{iθ}:V_{-β}:]"""

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters

    # distinguish between mS and sG model 
    if modelParameters isa MassiveSchwingerParameters
        θ = hamiltonianParameters[:θ]
        m = hamiltonianParameters[:m]
        M = hamiltonianParameters[:M]
        β = sqrt(4 * pi)    # β = sqrt(4 * pi) for the mS model 
    elseif modelParameters isa SineGordonParameters
        θ = 0.0    # default value for sG is 0
        m = 1.0    # m is the fermion mass of the fermionic equivalent model, which for the sG model is the soliton mass with is 1
        M = 0.0    # M is the boson mass used to define the harmonic mode frequencies, which for the sG model is 0
        β = hamiltonianParameters[:β]
    end
    L = hamiltonianParameters[:L]

    # get momentumModes
    momentumModes = modeOccupations[1, :]

    # get number of momentum modes
    numSites = length(momentumModes)

    # compute :V_β: and :V_{-β}:
    localOperators_neg = Vector{TensorMap{ComplexF64}}(undef, numSites)
    localOperators_pos = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for (siteIdx, momentumVal) in enumerate(momentumModes)
        physVecSpace = physSpaces[siteIdx]
        localOperators_neg[siteIdx] = localVertexOp(modelParameters, physVecSpace,
                                                    momentumVal, -1, β, M, L)
        localOperators_pos[siteIdx] = localVertexOp(modelParameters, physVecSpace,
                                                    momentumVal, +1, β, M, L)
    end

    expOperator_neg = SparseMPO(localOperators_neg)
    expOperator_pos = SparseMPO(localOperators_pos)

    # construct momentum-preserving MPO using a kroneckerDelta MPS
    kronDeltaSpaces = [space(localOp, 3)' for localOp in expOperator_neg]
    kroneckerDeltaMPS = generateKroneckerDeltaMPS(kronDeltaSpaces)

    # combined with kroneckerDeltaMPS to form full MPO 
    V_neg = 1 / 2 *
            convertLocalOperatorsToMPO(expOperator_neg, kroneckerDeltaMPS)
    V_pos = 1 / 2 *
            convertLocalOperatorsToMPO(expOperator_pos, kroneckerDeltaMPS)

    V_neg *= exp(+1im * θ)
    V_pos *= exp(-1im * θ)

    mpo_H1 = V_neg + V_pos

    return mpo_H1
end

function local_number_operators(Model::Union{MassiveSchwingerModel,
                                             SineGordonModel})

    # get modeOccupations and physSpaces
    modeOccupations = Model.modeOccupations
    physSpaces = Model.physSpaces

    # set number operator for every site
    numberOperators = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces))
    for (siteIdx, maxOccupation) in enumerate(modeOccupations[2, :])
        # get physical vector space
        physSpace = physSpaces[siteIdx]

        # set individual operators
        numberOperator = getNumberOperator(maxOccupation)
        if Model isa SineGordonModel
            if siteIdx == 1
                numberOperator = abs.(LinearAlgebra.diagm(collect((-maxOccupation):maxOccupation)))
            end
        end
        numberOperators[siteIdx] = TensorMap(numberOperator, physSpace,
                                             physSpace)
    end
    return numberOperators
end

function pairing_operators(Model::Union{MassiveSchwingerModel,SineGordonModel})

    #----------------------------------------------------------------------------
    # compute expectation values ⟨a(-k) a(+k)⟩ and ⟨a(-k)^† a(+k)^†⟩
    #----------------------------------------------------------------------------

    # get kMax
    kMax = Model.modelParameters.truncationParameters[:kMax]

    # get physSpaces and momentumModes
    physSpaces = Model.physSpaces
    momentumModes = Model.modeOccupations[1, :]

    # construct MPOs
    mpos_AnAn = Vector{SparseMPO}(undef, kMax)
    mpos_CrCr = Vector{SparseMPO}(undef, kMax)
    for (kIdx, kVal) in enumerate(collect(1:kMax))

        # construct momentum-preserving MPO for ⟨a(-k) a(+k)⟩
        localOperators = Vector{TensorMap{ComplexF64}}(undef,
                                                       length(physSpaces))
        for (siteIdx, momentumVal) in enumerate(momentumModes)
            if abs(momentumVal) == kVal
                localOperators[siteIdx] = +1im *
                                          localAnnihilationOp(momentumVal,
                                                              physSpaces[siteIdx])
            else
                localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx])
            end
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpos_AnAn[kIdx] = convertLocalOperatorsToMPO(localOperators)

        # construct momentum-preserving MPO for ⟨a(-k)^† a(+k)^†⟩
        localOperators = Vector{TensorMap{ComplexF64}}(undef,
                                                       length(physSpaces))
        for (siteIdx, momentumVal) in enumerate(momentumModes)
            if abs(momentumVal) == kVal
                localOperators[siteIdx] = -1im *
                                          localCreationOp(momentumVal,
                                                          physSpaces[siteIdx])
            else
                localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx])
            end
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpos_CrCr[kIdx] = convertLocalOperatorsToMPO(localOperators)
    end

    return mpos_AnAn, mpos_CrCr
end

function createDisplacementOp(nMax::Int64, z::Union{Int64,Float64,ComplexF64}, physSpace)
    """
    Construct the normal-ordered displacement operator: 
    : ̂D(z): = e^{z a^†} e^{-z^* a} 
    """
    Id = Diagonal(ones(ComplexF64, nMax + 1))
    An = -1im * diagm(+1 => sqrt.(collect(1:nMax)))
    getIdentityOperator(nMax + 1)
    getAnnihilationOperator(nMax)

    return idOp = TensorMap(getIdentityOperator(dim(physSpace)), physSpace, physSpace)
end
