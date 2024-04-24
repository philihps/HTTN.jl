
Pk_sW(k, L) = 2*pi/L * k;
Ek_sW(k, L, M) = sqrt(Pk_sW(k, L)^2 + M^2);

function energyMomentum_sW_massive(k::Union{Int64, Float64}, L::Union{Int64, Float64}, M::Union{Int64, Float64})
    """ Function to compute the energy corresponding to momentum k """

    Ek = sqrt(Pk_sW(k, L)^2 + M^2);
    return Ek;
end

function energyMomentum_sW_massless(k::Union{Int64, Float64}, L::Float64, M::Union{Int64, Float64})
    """ Function to compute the energy corresponding to momentum k """

    # Ek = (2*pi/L * abs(k) + 1/(4*pi) * M * M * L / abs(k));
    Ek = 2*pi/L * abs(k);
    return Ek;
end

function convertLocalOperatorsToMPO(localOperators::Vector{<:AbstractTensorMap}; truncMomentumQNs::Union{Int64, Float64} = Inf)
    """ Generates MPO out of three-index local operators using kroneckerDeltaMPS """

    # construct kroneckerDeltaMPS
    kronDeltaIndices = [space(localOp, 3)' for localOp in localOperators];
    kroneckerDeltaMPS = generateKroneckerDeltaMPS(kronDeltaIndices);

    # combine local operators and kroneckerDeltaMPS
    localOperators = SparseEXP(localOperators);
    finalMPO = convertLocalOperatorsToMPO(localOperators, kroneckerDeltaMPS);
    return finalMPO;
    
end

function convertLocalOperatorsToMPO(localOperators::SparseEXP, kroneckerDeltaMPS::SparseMPS)
    """ Generates MPO out of three-index local operators using kroneckerDeltaMPS """

    # get length of localOperators
    numSites = length(localOperators);

    # combine local operators and kroneckerDeltaMPS
    mpoTensors = Vector{TensorMap}(undef, numSites);
    for siteIdx = 1 : numSites
        @tensor mpoTensors[siteIdx][-1 -2; -3 -4] := localOperators[siteIdx][-2, -4, 1] * kroneckerDeltaMPS[siteIdx][-1, 1, -3];
    end
    return SparseMPO(mpoTensors);

end

function constructIdentiyMPO(physSpaces::Vector{<:Union{ElementarySpace, CompositeSpace{ElementarySpace}}}, virtVecSpace::Union{ElementarySpace, CompositeSpace{ElementarySpace}})
    """ Constructs the identity MPO with trivial vector space on virtual indices """

    # construct identity MPOs
    identityMPO = Vector{TensorMap}(undef, length(physSpaces));
    for (siteIdx, physSpace) = enumerate(physSpaces)
        dimPhysSpace = dim(physSpace);
        localMPO  = zeros(Float64, dim(virtVecSpace), dim(physSpace), dim(virtVecSpace), dim(physSpace));
        localMPO[1, :, 1, :] = diagm(ones(dimPhysSpace));
        identityMPO[siteIdx] = TensorMap(localMPO, virtVecSpace ⊗ physSpace, virtVecSpace ⊗ physSpace);
    end
    return SparseMPO(identityMPO);

end