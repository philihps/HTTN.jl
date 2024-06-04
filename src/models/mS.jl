
struct MassiveSchwingerParameters <: AbstractQFTModelParameters
    
    truncationParameters::NamedTuple
    hamiltonianParameters::NamedTuple

    function MassiveSchwingerParameters(truncationParameters::NamedTuple, hamiltonianParameters::NamedTuple)
        return new(truncationParameters, hamiltonianParameters)
    end

end

struct MassiveSchwingerModel <: AbstractQFTModel
    modelParameters::MassiveSchwingerParameters
    modeOccupations::Matrix{Int64}
    physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}}
    # mpo_H0::SparseMPO
    # mpo_H1::SparseMPO
    # modelMPO::SparseMPO
end



function MassiveSchwingerModel(truncationParameters::NamedTuple, hamiltonianParameters::NamedTuple)
    
    # construct struct to store all model parameters
    modelParameters = MassiveSchwingerParameters(truncationParameters, hamiltonianParameters);

    # get hamiltonianParameters
    θ = hamiltonianParameters[:θ];
    m = hamiltonianParameters[:m];
    M = hamiltonianParameters[:M];
    L = hamiltonianParameters[:L];

    # construct modeOccupations and physSpaces
    modeOccupations, physSpaces = constructPhysSpaces(modelParameters);
    # display(modeOccupations)
    # display(physSpaces)

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
    return MassiveSchwingerModel(modelParameters, modeOccupations, physSpaces);

end

function updateBogoliubovPrameters(mSModel::MassiveSchwingerModel, bogParameters::Union{Vector{Int64}, Vector{Float64}, Vector{ComplexF64}})

    # update truncationParameters
    truncationParameters = mSModel.modelParameters.truncationParameters;
    truncationParameters = (
        kMax = truncationParameters[:kMax], 
        nMax = truncationParameters[:nMax], 
        nMaxZM = truncationParameters[:nMaxZM], 
        truncMethod = truncationParameters[:truncMethod], 
        modeOrdering = truncationParameters[:modeOrdering], 
        bogoliubovR = truncationParameters[:bogoliubovR], 
        bogParameters = bogParameters
    );

    # get hamiltonianParameters
    hamiltonianParameters = mSModel.modelParameters.hamiltonianParameters;

    # construct struct to store all model parameters
    modelParameters = MassiveSchwingerParameters(truncationParameters, hamiltonianParameters);
    return MassiveSchwingerModel(modelParameters, mSModel.modeOccupations, mSModel.physSpaces);

end

function generate_H0(mSModel::MassiveSchwingerModel)

    mpo_H0 = generate_H0(mSModel.modelParameters, mSModel.modeOccupations, mSModel.physSpaces);
    return mpo_H0;

end

function generate_H1(mSModel::MassiveSchwingerModel)

    modeOccupations, physSpaces = constructPhysSpaces(mSModel.modelParameters);
    mpo_H1 = generate_H1(modelParameters, modeOccupations, physSpaces);
    return mpo_H1;

end

function generate_MPO_mS(mSModel::MassiveSchwingerModel)

    # get hamiltonianParameters
    θ = mSModel.modelParameters.hamiltonianParameters[:θ];
    m = mSModel.modelParameters.hamiltonianParameters[:m];
    M = mSModel.modelParameters.hamiltonianParameters[:M];
    L = mSModel.modelParameters.hamiltonianParameters[:L];
    
    # construct MPOs
    modeOccupations, physSpaces = constructPhysSpaces(mSModel.modelParameters);
    mpo_H0 = generate_H0(mSModel.modelParameters, modeOccupations, physSpaces);
    mpo_H1 = generate_H1(mSModel.modelParameters, modeOccupations, physSpaces);
    
    # apply Hamiltonian prefactors and add mpo_H0 and mpo_H1
    if m == 0
        mpo_mS = mpo_H0;
    else

        # use infinite size prefactor for H1
        convFactor = - L * m * M / (4 * pi) * exp(0.5772156649);
        mpo_H1 *= convFactor;

        # sum up H0 and H1 with corresponding prefactors
        mpo_mS = mpo_H0 + mpo_H1;

    end
    return mpo_mS;

end

function modelSetup(modelParameters::MassiveSchwingerParameters)

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    kMax = truncationParameters[:kMax];
    nMax = truncationParameters[:nMax];
    nMaxZM = truncationParameters[:nMaxZM];
    truncMethod = truncationParameters[:truncMethod];
    modeOrdering = truncationParameters[:modeOrdering];
    bogoliubovR = truncationParameters[:bogoliubovR];

    # set momentum values
    momentumModes = convert.(Int64, collect(-kMax : 1 : +kMax));
    numSites = length(momentumModes);

    # sort momentumModes according to abs value
    if modeOrdering == 1
        momentumModes = sort(momentumModes, by = abs);
    end

    # set maximal occupation number and cutOff for maximal momentum in the physical modes
    if truncMethod == 1 || truncMethod == 4
        nMax = 1 * kMax;
        maximalMomentumPhysicalModes = 1 * kMax;
    elseif truncMethod == 2
        nMax = 2 * kMax;
        maximalMomentumPhysicalModes = 2 * kMax;
    elseif truncMethod == 5 || truncMethod == 6
        nMax = nMax;
        maximalMomentumPhysicalModes = nMax;
    end

    # set occupation of physical sites
    occPerSite = zeros(Int64, numSites);
    for (siteIdx, momentumVal) in enumerate(momentumModes)
        if momentumVal == 0
            occPerSite[siteIdx] = nMaxZM;
        else
            if truncMethod == 1 || truncMethod == 2 || truncMethod == 4 || truncMethod == 5 || truncMethod == 5
                occPerSite[siteIdx] = min(nMax, Int(floor(abs(maximalMomentumPhysicalModes / momentumVal))));
            elseif truncMethod == 3
                occPerSite[siteIdx] = nMax;
            end
        end
    end
    if truncMethod == 6
        occPerSite = calculateMaximalOccupations(truncMethod, kMax);
    end

    # combine momentum modes and maximal mode occupation numbers
    modeOccupations = vcat(reshape(momentumModes, 1, numSites), reshape(occPerSite, 1, numSites));
    return modeOccupations;

end

function constructPhysSpaces(modelParameters::MassiveSchwingerParameters)
    """ Constructs vector spaces for physical bond indices of the MPS """

    # construct modeOccupations
    modeOccupations = modelSetup(modelParameters);
    momentumModes = modeOccupations[1, :];
    occPerSite = modeOccupations[2, :];

    # get number of momentum modes
    numSites = length(momentumModes);

    # determine physSpaces
    physSpaces = Vector{typeof(U1Space())}(undef, numSites);
    for siteIdx = 1 : numSites
        momentumVal = momentumModes[siteIdx];
        maxOccupation = occPerSite[siteIdx];
        if momentumVal == 0
            physSpaces[siteIdx] = U1Space(0 => (maxOccupation + 1));
        else
            physSpaces[siteIdx] = U1Space([momentumVal * occup => 1 for occup = 0 : +maxOccupation]);
        end
    end
    return modeOccupations, physSpaces

end

function generate_H0_Part_A(modelParameters::MassiveSchwingerParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    
    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    bogoliubovR = truncationParameters[:bogoliubovR];

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters;
    θ = hamiltonianParameters[:θ];
    m = hamiltonianParameters[:m];
    M = hamiltonianParameters[:M];
    L = hamiltonianParameters[:L];

    # get momentumModes
    momentumModes = modeOccupations[1, :];
    numSites = length(physSpaces);

    # construct H0 Part A
    mpo_H0_Part_A = Vector{TensorMap}(undef, numSites);
    if numSites == 1

        for (siteIdx, momentumVal) in enumerate(momentumModes)

            # get physical vector space
            physVecSpace = physSpaces[siteIdx];
            dimHS = dim(physVecSpace);

            if momentumVal == 0

                # set modeFactor
                modeFactor = M;

                mpoBlock = zeros(Float64, 1, dimHS, 1, dimHS);
                mpoBlock[1, :, 1, :] = modeFactor * getNumberOperator(dimHS - 1);

            end

            # convert mpoBlock to TensorMap
            mpo_H0_Part_A[siteIdx] = TensorMap(mpoBlock, U1Space(0 => size(mpoBlock, 1)), physSpaces[siteIdx], U1Space(0 => size(mpoBlock, 3)) ⊗ physSpaces[siteIdx]);

        end

    else

        for (siteIdx, momentumVal) in enumerate(momentumModes)

            # get physical vector space
            physVecSpace = physSpaces[siteIdx];
            dimHS = dim(physVecSpace);

            if momentumVal == 0

                # set modeFactor
                modeFactor = M;

                if siteIdx == 1
                    mpoBlock = zeros(Float64, 1, dimHS, 2, dimHS);
                    mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS);
                    mpoBlock[1, :, 2, :] = modeFactor * getNumberOperator(dimHS - 1);
                elseif siteIdx == numSites
                    mpoBlock = zeros(Float64, 2, dimHS, 1, dimHS);
                    mpoBlock[1, :, 1, :] = modeFactor * getNumberOperator(dimHS - 1);
                    mpoBlock[2, :, 1, :] = getIdentityOperator(dimHS);
                else
                    mpoBlock = zeros(Float64, 2, dimHS, 2, dimHS);
                    mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS);
                    mpoBlock[1, :, 2, :] = modeFactor * getNumberOperator(dimHS - 1);
                    mpoBlock[2, :, 2, :] = getIdentityOperator(dimHS);
                end

            else

                # get ordering of QNs in physVecSpace
                qnSectors = physVecSpace.dims;
                phyVecSpaceOrdering = [productSector.charge for productSector in keys(qnSectors)];

                # set modeFactor
                if bogoliubovR == 0
                    modeFactor = energyMomentum_sW_massive(momentumVal, L, M);
                elseif bogoliubovR == 1
                    modeFactor = energyMomentum_sW_massless(momentumVal, L, M);
                end
                
                if siteIdx == 1
                    mpoBlock = zeros(Float64, 1, dimHS, 2, dimHS);
                    mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS);
                    numberOperator = diagm(abs.(phyVecSpaceOrdering ./ momentumVal));
                    mpoBlock[1, :, 2, :] = modeFactor * numberOperator;
                elseif siteIdx == numSites
                    mpoBlock = zeros(Float64, 2, dimHS, 1, dimHS);
                    numberOperator = diagm(abs.(phyVecSpaceOrdering ./ momentumVal));
                    mpoBlock[1, :, 1, :] = modeFactor * numberOperator;
                    mpoBlock[2, :, 1, :] = getIdentityOperator(dimHS);
                else
                    mpoBlock = zeros(Float64, 2, dimHS, 2, dimHS);
                    mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS);
                    numberOperator = diagm(abs.(phyVecSpaceOrdering ./ momentumVal));
                    mpoBlock[1, :, 2, :] = modeFactor * numberOperator;
                    mpoBlock[2, :, 2, :] = getIdentityOperator(dimHS);
                end

            end
            
            # convert mpoBlock to TensorMap
            mpo_H0_Part_A[siteIdx] = TensorMap(mpoBlock, U1Space(0 => size(mpoBlock, 1)) ⊗ physSpaces[siteIdx], U1Space(0 => size(mpoBlock, 3)) ⊗ physSpaces[siteIdx]);

        end

    end
    return SparseMPO(mpo_H0_Part_A);

end

function generate_H0(modelParameters::MassiveSchwingerParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    """ Function to generate the non-interacting part H0 of the massive Schwinger Hamiltonian """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    bogoliubovR = truncationParameters[:bogoliubovR];

    # get diagonal part of H0 (part A)
    mpo_H0 = generate_H0_Part_A(modelParameters, modeOccupations, physSpaces);

    # get additional parts due to the Bogoliubov rotation
    if bogoliubovR == 1
    #     mpo_H0_B = generate_H0_sW_Part_B(momentumModes, physSpaces, M, L);
    #     mpo_H0 = addMPOs(mpo_H0, mpo_H0_B);
    #     mpo_H0_C = generate_H0_sW_Part_C(momentumModes, physSpaces, M, L);
    #     mpo_H0 = addMPOs(mpo_H0, mpo_H0_C);
    end
    
    # # remove constant contribution of -2πL/12
    # vacVecSpace = Rep[U₁](0 => 1);
    # idMPO = constructIdentiyMPO(physSpaces, vacVecSpace);
    # idMPO[chainCenter] *= -2π * L /12;
    # mpo_H0 = addMPOs(mpo_H0, idMPO);
    return mpo_H0;

end

function F_mS(nBra::Int64, nKet::Int64, j::Int64, k::Union{Int64, Float64}, alpha::Union{Int64, Float64}, L::Float64, M::Union{Int64, Float64})::Float64
    """ Matrix element of the k-factor of V_{(±1, 0)}(x = 0, t = 0) for compactification radius R=sqrt(4π) in a massive mode basis of mass M """
    
    matEL = convert(Float64, (-1)^(j + nKet - nBra) * (alpha / sqrt(2 * energyMomentum_sW_massive(k, L, M) * L))^(2*j + nKet - nBra) * sqrt(factorial(big(nBra)) * factorial(big(nKet))) / factorial(big(j)) / factorial(big(nBra - j)) / factorial(big(j + nKet - nBra)));
    return matEL
end

function localVertexOp_mS(momentumVal::Int64, physVecSpace::Union{ElementarySpace, CompositeSpace{ElementarySpace}}, a::Float64, M::Float64, L::Float64, bogoliubovR::Int64 = 0)
    """ Construct local vertex operator, to be combined with kroneckerDeltaMPS to form full MPO """

    # construct kroneckerDelta space
    kronDelSpace = removeDegeneracyQN(fuse(physVecSpace, conj(flip(physVecSpace))));
    
    # get dimensions of physVecSpace and kronDelSpace
    dimPhyVecSpace = dim(physVecSpace);
    dimAuxVecSpace = dim(kronDelSpace);

    # compute vertex operator coefficient α
    if bogoliubovR == 0
        alpha = a * sqrt(4 * pi);
    elseif bogoliubovR == 1
        alpha = a;
    end

    # construct local interaction for k = 0 or k ≂̸ 0
    if momentumVal == 0

        # fill interactionTensor
        interactionTensor = zeros(Float64, dimPhyVecSpace, dimPhyVecSpace, dimAuxVecSpace);
        for nBra = 0 : (dimPhyVecSpace - 1), nKet = 0 : (dimPhyVecSpace - 1)
            interactionTensor[nBra + 1, nKet + 1, 1] = convert(Float64, sum([F_mS(nBra, nKet, j, momentumVal, alpha, L, M) for j = max(0, nBra - nKet) : nBra]));;
        end
        
    else

        # get ordering of QNs in physVecSpace and kronDelSpace
        phyQNSectors = physVecSpace.dims;
        auxQNSectors = kronDelSpace.dims;
        phyVecSpaceOrdering = [productSector.charge for productSector in keys(phyQNSectors)];
        auxVecSpaceOrdering = [productSector.charge for productSector in keys(auxQNSectors)];
        
        # fill interactionTensor
        interactionTensor = zeros(Float64, dimPhyVecSpace, dimPhyVecSpace, dimAuxVecSpace);
        for nBra = 0 : (dimPhyVecSpace - 1), nKet = 0 : (dimPhyVecSpace - 1)
            braIndPos = findfirst(phyVecSpaceOrdering .== (momentumVal * nBra));
            ketIndPos = findfirst(phyVecSpaceOrdering .== (momentumVal * nKet));
            auxIndPos = findfirst(auxVecSpaceOrdering .== (momentumVal * (nBra - nKet)));
            interactionTensor[braIndPos, ketIndPos, auxIndPos] = convert(Float64, sum([F_mS(nBra, nKet, j, momentumVal, alpha, L, M) for j = max(0, nBra - nKet) : nBra]));
        end

        # # add factor for the Bogoliubov rotation
        # if bogoliubovR == 1
        #     interactionTensor *= exp(1 / abs(momentumVal) * (1 - Pk_sW(momentumVal, L) / Ek_sW(momentumVal, L, M)));
        # end

    end

    # convert interactionTensor to TensorMap with U1Space
    return TensorMap(interactionTensor, physVecSpace, physVecSpace ⊗ kronDelSpace);

end

function generate_H1(modelParameters::MassiveSchwingerParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    """ Function to generate the interacting part H1 of the Schwinger Hamiltonian """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    kMax = truncationParameters[:kMax];
    nMax = truncationParameters[:nMax];
    nMaxZM = truncationParameters[:nMaxZM];
    truncMethod = truncationParameters[:truncMethod];
    modeOrdering = truncationParameters[:modeOrdering];
    bogoliubovR = truncationParameters[:bogoliubovR];

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters;
    θ = hamiltonianParameters[:θ];
    m = hamiltonianParameters[:m];
    M = hamiltonianParameters[:M];
    L = hamiltonianParameters[:L];

    # get momentumModes
    momentumModes = modeOccupations[1, :];

    # get number of momentum modes
    numSites = length(momentumModes);
    chainCenter = Int((numSites - 1) / 2 + 1);

    # construct local vertex operators V_{-1, 0}(0, 0) and V_{+1, 0}(0, 0)
    localOperators_neg = Vector{TensorMap}(undef, numSites);
    localOperators_pos = Vector{TensorMap}(undef, numSites);
    for (siteIdx, momentumVal) in enumerate(momentumModes)
        physVecSpace = physSpaces[siteIdx];
        localOperators_neg[siteIdx] = localVertexOp_mS(momentumVal, physVecSpace, -1.0, M, L, bogoliubovR);
        localOperators_pos[siteIdx] = localVertexOp_mS(momentumVal, physVecSpace, +1.0, M, L, bogoliubovR);
    end
    expOperator_neg = SparseEXP(localOperators_neg);
    expOperator_pos = SparseEXP(localOperators_pos);

    # construct momentum-preserving MPO using a kroneckerDelta MPS
    kronDeltaSpaces = [space(localOp, 3)' for localOp in expOperator_neg];
    kroneckerDeltaMPS = generateKroneckerDeltaMPS(kronDeltaSpaces);
    V_neg = convertLocalOperatorsToMPO(expOperator_neg, kroneckerDeltaMPS);
    V_pos = convertLocalOperatorsToMPO(expOperator_pos, kroneckerDeltaMPS);

    # add factor of 1/2 due to cos(x) = 1/2 * (exp(+ix) + exp(-ix))
    if θ != 0
        V_neg *= exp(-1im * θ);
        V_pos *= exp(+1im * θ);
    end

    # construct mpo_H1
    mpo_H1 = V_neg + V_pos;
    if (θ == 0) || (θ == π)
        mpo_H1 = real.(mpo_H1);
    end

    # # add correct factor to convert between massive and massless basis
    # kMax = momentumModes[end];
    # if bogoliubovR == 0

    #     # # convertion factor k → ∞ and for L → ∞
    #     # gamma = 0.5772156649;
    #     # convFactor = - m * M / (4*pi) * exp(gamma);
        
    #     # # convertion factor k → ∞ and for finite L
    #     # sumTerms = Float64[];
    #     # for kVal = 1 : kMax
    #     #     sumTerms = vcat(sumTerms, 1 / kVal * (1 - Pk_sW(kVal, L) / Ek_sW(kVal, L, M)));
    #     # end
    #     # convFactor = - m * exp(sum(sumTerms));

    #     # convertion factor k → ∞ and for finite L
    #     kVal = 1;
    #     sumTerms = [1 / kVal * (1 - Pk_sW(kVal, L) / Ek_sW(kVal, L, M))];
    #     while abs(sumTerms[end]) > 1e-8
    #         kVal += 1;
    #         sumTerms = vcat(sumTerms, 1 / kVal * (1 - Pk_sW(kVal, L) / Ek_sW(kVal, L, M)));
    #     end
    #     convFactor = - m * exp(sum(sumTerms));

    #     # multiply H1
    #     mpo_H1[chainCenter] *= convFactor;

    # elseif bogoliubovR == 1
    #     # convFactor = exp(sum([1 / k * (1 - Pk_sW(k, L) / Ek_sW(k, L, M)) for k = (0 + 1) : 100]));
    #     mpo_H1[chainCenter] *= convFactor;
    # end
    return mpo_H1;

end

function local_number_operators(mS::MassiveSchwingerModel)

    # get modeOccupations and physSpaces
    modeOccupations = mS.modeOccupations;
    physSpaces = mS.physSpaces;

    # set number operator for every site
    numberOperators = Vector{TensorMap}(undef, length(physSpaces));
    for (siteIdx, maxOccupation) in enumerate(modeOccupations[2, :])

        # get physical vector space
        physSpace = physSpaces[siteIdx];

        # set individual operators
        numberOperator = getNumberOperator(maxOccupation);
        numberOperators[siteIdx] = TensorMap(numberOperator, physSpace, physSpace);

    end
    return numberOperators

end

function pairing_operators(mS::MassiveSchwingerModel)

    #----------------------------------------------------------------------------
    # compute expectation values ⟨a(-k) a(+k)⟩ and ⟨a(-k)^† a(+k)^†⟩
    #----------------------------------------------------------------------------

    # get kMax
    kMax = mS.modelParameters.truncationParameters[:kMax];

    # get physSpaces and momentumModes
    physSpaces = mS.physSpaces;
    momentumModes = mS.modeOccupations[1, :];

    # construct MPOs
    mpos_AnAn = Vector{SparseMPO}(undef, kMax);
    mpos_CrCr = Vector{SparseMPO}(undef, kMax);
    for (kIdx, kVal) = enumerate(collect(1 : kMax))

        # construct momentum-preserving MPO for ⟨a(-k) a(+k)⟩
        localOperators = Vector{TensorMap}(undef, length(physSpaces));
        for (siteIdx, momentumVal) = enumerate(momentumModes)
            if abs(momentumVal) == kVal
                localOperators[siteIdx] = +1im * localAnnihilationOp(momentumVal, physSpaces[siteIdx]);
            else
                localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx]);
            end
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpos_AnAn[kIdx] = convertLocalOperatorsToMPO(localOperators);

        # construct momentum-preserving MPO for ⟨a(-k)^† a(+k)^†⟩
        localOperators = Vector{TensorMap}(undef, length(physSpaces));
        for (siteIdx, momentumVal) = enumerate(momentumModes)
            if abs(momentumVal) == kVal
                localOperators[siteIdx] = -1im * localCreationOp(momentumVal, physSpaces[siteIdx]);
            else
                localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx]);
            end
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpos_CrCr[kIdx] = convertLocalOperatorsToMPO(localOperators);

    end

    return mpos_AnAn, mpos_CrCr

end