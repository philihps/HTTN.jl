
function convertLocalOperatorsToMPO(localOperators::Vector{<:AbstractTensorMap};
                                    truncMomentumQNs::Union{Int64,Float64} = Inf,)
    """ Generates MPO out of three-index local operators using kroneckerDeltaMPS """

    # construct kroneckerDeltaMPS
    kronDeltaIndices = [space(localOp, 3)' for localOp in localOperators]
    kroneckerDeltaMPS = generateKroneckerDeltaMPS(kronDeltaIndices)

    # combine local operators and kroneckerDeltaMPS
    localOperators = SparseMPO(localOperators)
    finalMPO = convertLocalOperatorsToMPO(localOperators, kroneckerDeltaMPS)
    return finalMPO
end

function convertLocalOperatorsToMPO(localOperators::SparseMPO, kroneckerDeltaMPS::SparseMPS)
    """ Generates MPO out of three-index local operators using kroneckerDeltaMPS """

    # get length of localOperators
    numSites = length(localOperators)

    # combine local operators and kroneckerDeltaMPS
    mpoTensors = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for siteIdx in 1:numSites
        @tensor mpoTensors[siteIdx][-1 -2; -3 -4] := localOperators[siteIdx][-2, -4, 1] *
                                                     kroneckerDeltaMPS[siteIdx][-1, 1, -3]
    end
    return SparseMPO(mpoTensors)
end

function constructIdentityMPO(physSpaces::Vector{<:Union{ElementarySpace,
                                                         CompositeSpace{ElementarySpace}}},
                              virtVecSpace::Union{ElementarySpace,
                                                  CompositeSpace{ElementarySpace}})
    """ Constructs the identity MPO with trivial vector space on virtual indices """

    # construct identity MPOs
    identityMPO = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces))
    for (siteIdx, physSpace) in enumerate(physSpaces)
        dimPhysSpace = dim(physSpace)
        localMPO = zeros(ComplexF64, dim(virtVecSpace), dim(physSpace), dim(virtVecSpace),
                         dim(physSpace))
        localMPO[1, :, 1, :] = diagm(ones(dimPhysSpace))
        identityMPO[siteIdx] = TensorMap(localMPO, virtVecSpace ⊗ physSpace,
                                         virtVecSpace ⊗ physSpace)
    end
    return SparseMPO(identityMPO)
end
