
function convertLocalOperatorsToMPO(localOperators::Vector{<:AbstractTensorMap};
                                    truncMomentumQNs::Union{Int64,Float64} = Inf,)
    """ Generates MPO out of three-index local operators using kroneckerDeltaMPS """

    # construct kroneckerDeltaMPS
    kronDeltaIndices = [space(localOp, 3)' for localOp in localOperators]
    kroneckerDeltaMPS = generateKroneckerDeltaMPS(kronDeltaIndices)

    # combine local operators and kroneckerDeltaMPS
    localOperators = SparseEXP(localOperators)
    finalMPO = convertLocalOperatorsToMPO(localOperators, kroneckerDeltaMPS)
    return finalMPO
end

function convertLocalOperatorsToMPO(localOperators::SparseEXP, kroneckerDeltaMPS::SparseMPS)
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

function mps2vec(finiteMPS)
    N = length(finiteMPS)
    qnL = space(finiteMPS[1], 1)
    qnR = space(finiteMPS[N], 3)
    boundaryL = TensorMap(ones, one(qnL), qnL)
    boundaryR = TensorMap(ones, qnR', one(qnR))
    indexList = Vector{Vector{Int}}(undef, N + 2)
    for idxT in eachindex(indexList)
        if idxT == 1
            indexList[idxT] = [idxT]
        elseif idxT == length(indexList)
            indexList[idxT] = [idxT - 1]
        else
            indexList[idxT] = [idxT - 1, -(idxT - 1), idxT]
        end
    end
    mpsTensor = ncon(vcat(boundaryL, finiteMPS..., boundaryR), indexList)
    mpsTensor = convert(Array, mpsTensor)
    # sizeMPSTensor = size(mpsTensor)
    # mpsVector = reshape(mpsTensor, prod(sizeMPSTensor[1:N]))
    return mpsTensor
end

function mpo2mat(finiteMPO)
    N = length(finiteMPO)
    qnL = space(finiteMPO[1], 1)
    qnR = space(finiteMPO[N], 3)
    boundaryL = TensorMap(ones, one(qnL), qnL)
    boundaryR = TensorMap(ones, qnR', one(qnR))
    indexList = Vector{Vector{Int}}(undef, N + 2)
    for idxT in eachindex(indexList)
        if idxT == 1
            indexList[idxT] = [idxT]
        elseif idxT == length(indexList)
            indexList[idxT] = [idxT - 1]
        else
            indexList[idxT] = [idxT - 1, -(idxT - 1), idxT, -(idxT - 1) - N]
        end
    end
    mpoTensor = ncon(vcat(boundaryL, finiteMPO..., boundaryR), indexList)
    mpoTensor = permute(mpoTensor, Tuple(1:N), Tuple((N + 1):(2 * N)))
    mpoTensor = convert(Array, mpoTensor)
    # sizeMPOTensor = size(mpoTensor)
    # mpoMatrix = reshape(mpoTensor, prod(sizeMPOTensor[1:N]), prod(sizeMPOTensor[1:N]))
    return mpoTensor
end
