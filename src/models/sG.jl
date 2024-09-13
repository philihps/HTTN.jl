
struct SineGordonParameters <: AbstractQFTModelParameters
    
    truncationParameters::NamedTuple
    hamiltonianParameters::NamedTuple

    function SineGordonParameters(truncationParameters::NamedTuple, hamiltonianParameters::NamedTuple)
        return new(truncationParameters, hamiltonianParameters)
    end

end

struct SineGordonModel <: AbstractQFTModel
    modelParameters::SineGordonParameters
    modeOccupations::Matrix{Int64}
    physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}}
    # mpo_H0::SparseMPO
    # mpo_H1::SparseMPO
    # modelMPO::SparseMPO
end

function getMomentumModes(sG::SineGordonModel)
    return sG.modeOccupations[1, :]
end

function kappaFunc(p::Float64)
    kappaVal = 2 * gamma(p^2 / 2) / pi / gamma(1 - p^2 / 2) * (sqrt(pi) * gamma(1 / (2 - p^2)) / 2 / gamma(p^2 / (4 - 2 * p^2)))^(2 - p^2);
    return kappaVal;
end

function SineGordonModel(truncationParameters::NamedTuple, hamiltonianParameters::NamedTuple)
    
    # construct struct to store all model parameters
    modelParameters = SineGordonParameters(truncationParameters, hamiltonianParameters);

    # get hamiltonianParameters
    β = hamiltonianParameters[:β];
    λ = hamiltonianParameters[:λ];
    L = hamiltonianParameters[:L];
    R = hamiltonianParameters[:R];

    # construct modeOccupations and physSpaces
    modeOccupations, physSpaces = constructPhysSpaces(modelParameters);
    # display(modeOccupations)
    # display(physSpaces)

    # # construct non-interacting MPO
    # mpo_H0 = generate_H0(modelParameters, modeOccupations, physSpaces);

    # # construct interacting MPO
    # mpo_H1 = generate_H1(modelParameters, modeOccupations, physSpaces);

    # # apply Hamiltonian prefactors and add mpo_H0 and mpo_H1
    # if λ == 0.0
    #     mpo_sG = 2 * π / L * mpo_H0;
    # else        
    #     p = 1.0 / R;
    #     convFactor = - λ * 2 * π / L * kappaFunc(p) * (L)^(2 - p^2) / (2 * pi)^(1 - p^2);
    #     mpo_sG = 2 * π / L * mpo_H0 + convFactor * mpo_H1;
    # end
    # return SineGordonModel(modelParameters, modeOccupations, physSpaces, mpo_H0, mpo_H1, mpo_sG);

    return SineGordonModel(modelParameters, modeOccupations, physSpaces);

end

function updateBogoliubovParameters(sGModel::SineGordonModel, bogParameters::Union{Vector{Int64}, Vector{Float64}, Vector{ComplexF64}})

    # update truncationParameters
    truncationParameters = sGModel.modelParameters.truncationParameters;
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
    hamiltonianParameters = sGModel.modelParameters.hamiltonianParameters;

    # construct struct to store all model parameters
    modelParameters = SineGordonParameters(truncationParameters, hamiltonianParameters);
    return SineGordonModel(modelParameters, sGModel.modeOccupations, sGModel.physSpaces);

end

function initializeVacuumMPS(sG::SineGordonModel; modeOrdering::Int64 = 1)
    """ construct vacuum MPS tensor with physSpaces and virtSpaces """

    # get physSpaces
    physSpaces = sG.physSpaces;
    numSites = length(physSpaces);

    # set virtSpaces
    virtSpaces = fill(U1Space(0 => 1), numSites + 1);

    # initialize vacuum MPS
    mpsTensors = Vector{TensorMap}(undef, numSites);
    if modeOrdering == 0
        zeroSitePos = Int((numSites - 1) / 2 + 1);
    elseif modeOrdering == 1
        zeroSitePos = 1;
    end
    for siteIdx = 1 : numSites
        if siteIdx == zeroSitePos
            mpsTensor = zeros(ComplexF64, dim(virtSpaces[siteIdx]), dim(physSpaces[siteIdx]), dim(virtSpaces[siteIdx + 1]));
            middleIndex = Int((dim(physSpaces[siteIdx]) - 1) / 2) + 1;
            mpsTensor[1, middleIndex, 1] = 1.0;
            mpsTensors[siteIdx] = TensorMap(mpsTensor, virtSpaces[siteIdx] ⊗ physSpaces[siteIdx], virtSpaces[siteIdx + 1]);
        else
            mpsTensors[siteIdx] = TensorMap(ones, ComplexF64, virtSpaces[siteIdx] ⊗ physSpaces[siteIdx], virtSpaces[siteIdx + 1]);
        end
    end
    return SparseMPS(mpsTensors, normalizeMPS = true);

end

function initializeMPS(sG::SineGordonModel, initMPS::SparseMPS; modeOrdering::Int64 = 1)
    """ construct random MPS tensor with physVecSpaces and virtVecSpaces """

    # construct physical and virtual vector spaces for the MPS
    physSpaces = sG.physSpaces;
    boundarySpaceL = U1Space(0 => 1);
    boundarySpaceR = U1Space(0 => 1);
    virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);
    
    numSites = length(physSpaces);
    mpsTensors = Vector{TensorMap}(undef, numSites);
    if modeOrdering == 0
        zeroSitePos = Int((numSites - 1) / 2 + 1);
    elseif modeOrdering == 1
        zeroSitePos = 1;
    end
    for siteIdx = 1 : numSites
        initTensor = 1e-0 * convert(Array, initMPS[siteIdx]);
        siteTensor = TensorMap(randn, Float64, virtSpaces[siteIdx] ⊗ physSpaces[siteIdx], virtSpaces[siteIdx + 1]);
        if siteIdx == zeroSitePos
            siteTensor = 1e-1 * convert(Array, siteTensor);
        else
            siteTensor = 1e-1 * convert(Array, siteTensor);
        end
        minTensorDim = min(size(initTensor), size(siteTensor));
        siteTensor[1 : minTensorDim[1], 1 : minTensorDim[2], 1 : minTensorDim[3]] += initTensor[1 : minTensorDim[1], 1 : minTensorDim[2], 1 : minTensorDim[3]];
        mpsTensors[siteIdx] = TensorMap(siteTensor, virtSpaces[siteIdx] ⊗ physSpaces[siteIdx], virtSpaces[siteIdx + 1]);
    end
    return SparseMPS(mpsTensors, normalizeMPS = true);

end

function generate_H0(sGModel::SineGordonModel)

    mpo_H0 = generate_H0(sGModel.modelParameters, sGModel.modeOccupations, sGModel.physSpaces);
    return mpo_H0;

end

function generate_H1(sGModel::SineGordonModel)

    modeOccupations, physSpaces = constructPhysSpaces(sGModel.modelParameters);
    mpo_H1 = generate_H1(modelParameters, modeOccupations, physSpaces);
    return mpo_H1;

end

function generate_MPO_sG(sGModel::SineGordonModel)

    # get hamiltonianParameters
    β = sGModel.modelParameters.hamiltonianParameters[:β];
    λ = sGModel.modelParameters.hamiltonianParameters[:λ];
    L = sGModel.modelParameters.hamiltonianParameters[:L];
    R = sGModel.modelParameters.hamiltonianParameters[:R];
    
    # construct MPOs
    modeOccupations, physSpaces = constructPhysSpaces(sGModel.modelParameters);
    mpo_H0 = generate_H0(sGModel.modelParameters, modeOccupations, physSpaces);
    mpo_H1 = generate_H1(sGModel.modelParameters, modeOccupations, physSpaces);
    
    # apply Hamiltonian prefactors and add mpo_H0 and mpo_H1
    if λ == 0
        mpo_sG = 2 * π / L * mpo_H0;
    else        
        p = 1.0 / R;
        convFactor = - λ * 2 * π / L * kappaFunc(p) * (L)^(2 - p^2) / (2 * pi)^(1 - p^2);
        mpo_sG = 2 * π / L * mpo_H0 + convFactor * mpo_H1;
    end
    return mpo_sG;

end

function modelSetup(modelParameters::SineGordonParameters)

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

function constructPhysSpaces(modelParameters::SineGordonParameters)
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
            physSpaces[siteIdx] = U1Space(0 => (2 * maxOccupation + 1));
        else
            physSpaces[siteIdx] = U1Space([momentumVal * occup => 1 for occup = 0 : +maxOccupation]);
        end
    end
    return modeOccupations, physSpaces

end

function generateOperatorΠ0(nMaxZM::Int64, R::Float64; returnTM::Bool = false)
    """ Construct Π0 mode operators acting on (2 * nMaxZM + 1)-dimensional Hilbert space """

    opPi0 = 1 / R * LinearAlgebra.diagm(collect(-nMaxZM : nMaxZM));
    if returnTM == true
        opPi0 = TensorMap(opPi0, ComplexSpace(2 * nMaxZM + 1), ComplexSpace(2 * nMaxZM + 1));
    end
    return opPi0;

end

function energyMomentum_sG(k::Union{Int64, Float64})
    """ Function to compute the energy corresponding to momentum k """

    c = 1.0;
    return c * abs(k);
end

function generate_H0_Part_A(modelParameters::SineGordonParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    
    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    bogoliubovR = truncationParameters[:bogoliubovR];
    bogParameters = truncationParameters[:bogParameters];

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters;
    R = hamiltonianParameters[:R];
    L = hamiltonianParameters[:L];

    # get momentumModes
    momentumModes = modeOccupations[1, :];
    numSites = length(physSpaces);

    # construct H0 part A
    mpo_H0_Part_A = Vector{TensorMap}(undef, numSites);
    if numSites == 1

        for (siteIdx, momentumVal) in enumerate(momentumModes)

            # get physical vector space
            physSpace = physSpaces[siteIdx];
            dimHS = dim(physSpace);

            if momentumVal == 0

                # compute occupation per site for the zeroMode
                occupPerSite = Int(0.5 * (dimHS - 1));

                mpoBlock = zeros(Float64, 1, dimHS, 1, dimHS);
                mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS);
                mpoBlock[1, :, 1, :] = 1.0 * generateOperatorΠ0(occupPerSite, R)^2;

            end

            # convert mpoBlock to TensorMap
            mpo_H0_Part_A[siteIdx] = TensorMap(mpoBlock, U1Space(0 => size(mpoBlock, 1)), physSpaces[siteIdx], U1Space(0 => size(mpoBlock, 3)) ⊗ physSpaces[siteIdx]);

        end

    else

        for (siteIdx, momentumVal) in enumerate(momentumModes)

            # get physical vector space
            physSpace = physSpaces[siteIdx];
            dimHS = dim(physSpace);

            if momentumVal == 0

                # compute occupation per site for the zeroMode
                occupPerSite = Int(0.5 * (dimHS - 1));
            
                if siteIdx == 1
                    mpoBlock = zeros(Float64, 1, dimHS, 2, dimHS);
                    mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS);
                    mpoBlock[1, :, 2, :] = 1.0 * generateOperatorΠ0(occupPerSite, R)^2;
                elseif siteIdx == numSites
                    mpoBlock = zeros(Float64, 2, dimHS, 1, dimHS);
                    mpoBlock[1, :, 1, :] = 1.0 * generateOperatorΠ0(occupPerSite, R)^2;
                    mpoBlock[2, :, 1, :] = getIdentityOperator(dimHS);
                else
                    mpoBlock = zeros(Float64, 2, dimHS, 2, dimHS);
                    mpoBlock[1, :, 1, :] = getIdentityOperator(dimHS);
                    mpoBlock[1, :, 2, :] = 1.0 * generateOperatorΠ0(occupPerSite, R)^2;
                    mpoBlock[2, :, 2, :] = getIdentityOperator(dimHS);
                end

            else

                # get ordering of QNs in physSpace
                qnSectors = physSpace.dims;
                phyVecSpaceOrdering = [productSector.charge for productSector in keys(qnSectors)];

                # set modeFactor
                modeFactor = energyMomentum_sG(momentumVal);

                # get Bogoliubov rotation parameters
                if bogoliubovR == 1
                    kIdx = abs(momentumVal);
                    ξ = bogParameters[kIdx];
                    μ = real(cosh(abs(ξ)));
                    ν = real(sinh(abs(ξ)));
                    modeFactor *= (μ^2 + ν^2);
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

function generate_H0_Part_B(modelParameters::SineGordonParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    
    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    kMax = truncationParameters[:kMax];
    bogoliubovR = truncationParameters[:bogoliubovR];
    bogParameters = truncationParameters[:bogParameters];

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters;
    R = hamiltonianParameters[:R];
    L = hamiltonianParameters[:L];

    # get momentumModes
    momentumModes = modeOccupations[1, :];
    numSites = length(physSpaces);

    # construct H0 part B
    storeIndividualMPOs = Vector{SparseMPO}(undef, kMax);
    for (kIdx, kVal) in enumerate(collect(1 : kMax))

        # construct momentum-preserving MPO for < a(-k) a(+k) >
        localOperators = Vector{TensorMap}(undef, numSites);
        for (siteIdx, momentumVal) = enumerate(momentumModes)
            if abs(momentumVal) == kVal
                localOperators[siteIdx] = localAnnihilationOp(momentumVal, physSpaces[siteIdx]);
            else
                localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx]);
            end
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpoAnAn = convertLocalOperatorsToMPO(localOperators);

        # construct momentum-preserving MPO for < a(-k)^† a(+k)^† >
        localOperators = Vector{TensorMap}(undef, numSites);
        for (siteIdx, momentumVal) = enumerate(momentumModes)
            if abs(momentumVal) == kVal
                localOperators[siteIdx] = localCreationOp(momentumVal, physSpaces[siteIdx]);
            else
                localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx]);
            end
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpoCrCr = convertLocalOperatorsToMPO(localOperators);

        # get Bogoliubov rotation parameters (this is not checked for complex ξ)
        ξ = bogParameters[abs(kVal)];
        μ = real(cosh(abs(ξ)));
        ν = real(sinh(abs(ξ)));
        mpoAnAn[1 + 2 * (kIdx - 1) + 1] *= energyMomentum_sG(kVal) * (-2 * μ * ν);
        mpoCrCr[1 + 2 * (kIdx - 1) + 1] *= energyMomentum_sG(kVal) * (-2 * μ * ν);

        # store sum of MPOs
        storeIndividualMPOs[kIdx] = mpoAnAn + mpoCrCr;

    end

    # add together all MPOs
    mpo_H0_Part_B = copy(storeIndividualMPOs[1]);
    for kIdx = 2 : length(storeIndividualMPOs)
        mpo_H0_Part_B += storeIndividualMPOs[kIdx];
    end
    return mpo_H0_Part_B;

end

function generate_H0_Part_C(modelParameters::SineGordonParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    
    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    kMax = truncationParameters[:kMax];
    bogoliubovR = truncationParameters[:bogoliubovR];
    bogParameters = truncationParameters[:bogParameters];

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters;
    R = hamiltonianParameters[:R];
    L = hamiltonianParameters[:L];

    # get momentumModes
    momentumModes = modeOccupations[1, :];
    numSites = length(physSpaces);

    # construct H0 Part C
    storeIndividualMPOs = Vector{SparseMPO}(undef, kMax);
    for (kIdx, kVal) in enumerate(collect(1 : kMax))

        # construct momentum-presering identity MPO (constant energy term after Bogoliubov rotation)
        localOperators = Vector{TensorMap}(undef, numSites);
        for siteIdx = 1 : numSites
            localOperators[siteIdx] = localIdentityOp(physSpaces[siteIdx]);
        end

        # construct momentum-preserving MPO using a kroneckerDelta MPS
        mpoIdId = convertLocalOperatorsToMPO(localOperators);

        # get Bogoliubov rotation parameters (this is not checked for complex ξ)
        ξ = bogParameters[abs(kVal)]
        μ = real(cosh(abs(ξ)));
        ν = real(sinh(abs(ξ)));
        mpoIdId[1 + 2 * (kIdx - 1) + 1] *= energyMomentum_sG(kVal) * 2 * ν^2;

        # store MPO
        storeIndividualMPOs[kIdx] = mpoIdId;

    end

    # add together all MPOs
    mpo_H0_Part_C = copy(storeIndividualMPOs[1]);
    for kIdx = 2 : length(storeIndividualMPOs)
        mpo_H0_Part_C += storeIndividualMPOs[kIdx];
    end        
    return mpo_H0_Part_C;

end

function generate_H0(modelParameters::SineGordonParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    """ Function to generate the non-interacting part H0 of the massive Schwinger Hamiltonian """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    bogoliubovR = truncationParameters[:bogoliubovR];

    # get diagonal part of H0
    mpo_H0 = generate_H0_Part_A(modelParameters, modeOccupations, physSpaces);

    # get off-diagonal and constant part when using a Bogoliubov rotation
    if bogoliubovR == 1
        mpo_H0 += generate_H0_Part_B(modelParameters, modeOccupations, physSpaces);
        mpo_H0 += generate_H0_Part_C(modelParameters, modeOccupations, physSpaces);
    end

    # remove constant contribution of -1/12
    mpo_H0 += -1/12 * constructIdentiyMPO(physSpaces, Rep[U₁](0 => 1));
    
    # return mpo_H0
    return mpo_H0;

end

function F_sG(nBra::Int64, nKet::Int64, j::Int64, k::Union{Int64, Float64}, alpha::Float64)::Float64
    """ Matrix element of the k-factor of V_{(a, 0)}(x = 0, t = 0) for a given compactification radius R """
    
    matEL = convert(Float64, (-1)^(j + nKet - nBra) * alpha^(2*j + nKet - nBra) * abs(k)^((nBra - nKet) / 2-j) * sqrt(factorial(big(nBra)) * factorial(big(nKet))) / factorial(big(j)) / factorial(big(nBra - j)) / factorial(big(j + nKet - nBra)));
    return matEL
end

function localVertexOp_sG(momentumVal::Int64, physSpace::Union{ElementarySpace, CompositeSpace{ElementarySpace}}, a::Float64, R::Float64; bogoliubovR::Int64 = 0, ξ::Union{Int64, Float64} = 0.0)
    """ Construct local vertex operator, to be combined with kroneckerDeltaMPS to form full MPO """

    # construct kroneckerDelta space
    kronDelSpace = removeDegeneracyQN(fuse(physSpace, conj(flip(physSpace))));
    
    # get dimensions of physSpace and kronDelSpace
    dimPhyVecSpace = dim(physSpace);
    dimAuxVecSpace = dim(kronDelSpace);

    # construct local interaction for k = 0 or k ≂̸ 0
    if momentumVal == 0

        # fill interactionTensor
        interactionTensor = zeros(Float64, dimPhyVecSpace, dimPhyVecSpace, dimAuxVecSpace);
        if a == 0
            for rk = 1 : dimPhyVecSpace
                interactionTensor[rk, rk, 1] = 1.0;
            end
        elseif a > 0
            for rk = 1 : (dimPhyVecSpace - 1)
                interactionTensor[(rk + 1), rk, 1] = 1.0;
            end
        elseif a < 0
            for rk = 1 : (dimPhyVecSpace - 1)
                interactionTensor[rk, (rk + 1), 1] = 1.0;
            end
        end
        
    else

        # get ordering of QNs in physSpace and kronDelSpace
        phyQNSectors = physSpace.dims;
        auxQNSectors = kronDelSpace.dims;
        phyVecSpaceOrdering = [productSector.charge for productSector in keys(phyQNSectors)];
        auxVecSpaceOrdering = [productSector.charge for productSector in keys(auxQNSectors)];

        # compute vertex operator coefficient α
        alpha = a / R;

        # get Bogoliubov coefficients
        μ = real(cosh(abs(ξ)));
        ν = real(sinh(abs(ξ)));
        
        # fill interactionTensor
        interactionTensor = zeros(Float64, dimPhyVecSpace, dimPhyVecSpace, dimAuxVecSpace);
        for nBra = 0 : (dimPhyVecSpace - 1), nKet = 0 : (dimPhyVecSpace - 1)
            braIndPos = findfirst(phyVecSpaceOrdering .== (momentumVal * nBra));
            ketIndPos = findfirst(phyVecSpaceOrdering .== (momentumVal * nKet));
            auxIndPos = findfirst(auxVecSpaceOrdering .== (momentumVal * (nBra - nKet)));
            if bogoliubovR == 0
                interactionTensor[braIndPos, ketIndPos, auxIndPos] = convert(Float64, sum([F_sG(nBra, nKet, j, momentumVal, alpha) for j = max(0, nBra - nKet) : nBra]));
            elseif bogoliubovR == 1
                interactionTensor[braIndPos, ketIndPos, auxIndPos] = convert(Float64, sum([F_sG(nBra, nKet, j, momentumVal, alpha * (μ - ν)) for j = max(0, nBra - nKet) : nBra]));
            end
        end

        # add factor for the Bogoliubov rotation
        if bogoliubovR == 1
            z = alpha / sqrt(2 * energyMomentum_sG(momentumVal));
            interactionTensor *= exp(2 * z^2 * (μ - ν) * ν);
            # interactionTensor *= exp(alpha^2 / abs(momentumVal) * (μ - ν) * ν);
        end

    end

    # convert interactionTensor to TensorMap with U1Space
    return TensorMap(interactionTensor, physSpace, physSpace ⊗ kronDelSpace);

end

function generate_H1(modelParameters::SineGordonParameters, modeOccupations::Matrix{Int64}, physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}})
    """ Function to generate the interacting part H1 of the Schwinger Hamiltonian """

    # get truncationParameters
    truncationParameters = modelParameters.truncationParameters;
    kMax = truncationParameters[:kMax];
    nMax = truncationParameters[:nMax];
    nMaxZM = truncationParameters[:nMaxZM];
    truncMethod = truncationParameters[:truncMethod];
    modeOrdering = truncationParameters[:modeOrdering];
    bogoliubovR = truncationParameters[:bogoliubovR];
    bogParameters = truncationParameters[:bogParameters];

    # get hamiltonianParameters
    hamiltonianParameters = modelParameters.hamiltonianParameters;
    R = hamiltonianParameters[:R];
    L = hamiltonianParameters[:L];

    # get momentumModes
    momentumModes = modeOccupations[1, :];
    numSites = length(physSpaces);
    
    # construct local vertex operators V_{-1, 0}(0, 0) and V_{+1, 0}(0, 0)
    localOperators_neg = Vector{TensorMap}(undef, numSites);
    localOperators_pos = Vector{TensorMap}(undef, numSites);
    for (siteIdx, momentumVal) in enumerate(momentumModes)
        physSpace = physSpaces[siteIdx];
        momentumVal == 0 ? bogParameter = 0.0 : bogParameter = bogParameters[abs(momentumVal)];
        localOperators_neg[siteIdx] = localVertexOp_sG(momentumVal, physSpace, -1.0, R, ξ = bogParameter, bogoliubovR = bogoliubovR);
        localOperators_pos[siteIdx] = localVertexOp_sG(momentumVal, physSpace, +1.0, R, ξ = bogParameter, bogoliubovR = bogoliubovR);
    end

    if bogoliubovR == 0
        localOperators_neg = real.(localOperators_neg);
        localOperators_pos = real.(localOperators_pos);
    end

    # convert to sparse exponential operator form
    expOperator_neg = SparseEXP(localOperators_neg);
    expOperator_pos = SparseEXP(localOperators_pos);

    # construct momentum-preserving MPO using a kroneckerDelta MPS
    kronDeltaSpaces = [space(localOp, 3)' for localOp in expOperator_neg];
    kroneckerDeltaMPS = generateKroneckerDeltaMPS(kronDeltaSpaces);
    V_neg = convertLocalOperatorsToMPO(expOperator_neg, kroneckerDeltaMPS);
    V_pos = convertLocalOperatorsToMPO(expOperator_pos, kroneckerDeltaMPS);

    # add factor of 1/2 due to cos(x) = 1/2 * (exp(+ia) + exp(-ia))
    V_neg *= 0.5;
    V_pos *= 0.5;
    
    # construct mpo_H1
    mpo_H1 = V_neg + V_pos;
    return mpo_H1;

end

function local_number_operators(sG::SineGordonModel)

    # get modeOccupations and physSpaces
    modeOccupations = sG.modeOccupations;
    physSpaces = sG.physSpaces;

    # get compactification radius
    R = sG.modelParameters.hamiltonianParameters[:R];

    # set number operator for every site
    numberOperators = Vector{TensorMap}(undef, length(physSpaces));
    for (siteIdx, momentumVal) in enumerate(modeOccupations[1, :])

        # get physical vector space and occupation number
        physSpace = physSpaces[siteIdx];
        nMax = modeOccupations[2, siteIdx];

        # set individual operators
        if momentumVal == 0
            onSiteOperator = generateOperatorΠ0(nMax, R);
        else
            onSiteOperator = getNumberOperator(nMax);
        end
        numberOperators[siteIdx] = TensorMap(onSiteOperator, physSpace, physSpace);

    end
    return numberOperators;

end

function pairing_operators(sG::SineGordonModel)

    #----------------------------------------------------------------------------
    # compute expectation values ⟨a(-k) a(+k)⟩ and ⟨a(-k)^† a(+k)^†⟩
    #----------------------------------------------------------------------------

    # get kMax
    kMax = sG.modelParameters.truncationParameters[:kMax];

    # get physSpaces and momentumModes
    physSpaces = sG.physSpaces;
    momentumModes = sG.modeOccupations[1, :];

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
