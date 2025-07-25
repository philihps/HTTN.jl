function update_MPSEnvL(mpsEnvL, mpsTensorK, mpsTensorB)
    @tensor newEL[-1; -2] := mpsEnvL[1, 3] * mpsTensorK[3, 2, -2] *
                             conj(mpsTensorB[1, 2, -1])
    return newEL
end

function update_MPSEnvR(mpsEnvR, mpsTensorK, mpsTensorB)
    @tensor newER[-1; -2] := mpsTensorK[-1, 2, 1] * conj(mpsTensorB[-2, 2, 3]) *
                             mpsEnvR[1, 3]
    return newER
end

function update_MPOEnvL(mpoEnvL, mpsTensorK, mpoTensor, mpsTensorB)
    @tensor newEL[-1; -2 -3] := mpoEnvL[1, 3, 5] *
                                mpsTensorK[5, 4, -3] *
                                mpoTensor[3, 2, -2, 4] *
                                conj(mpsTensorB[1, 2, -1])
    return newEL
end

function update_MPOEnvR(mpoEnvR, mpsTensorK, mpoTensor, mpsTensorB)
    @tensor newER[-1 -2; -3] := mpsTensorK[-1, 2, 1] *
                                mpoTensor[-2, 4, 3, 2] *
                                conj(mpsTensorB[-3, 4, 5]) *
                                mpoEnvR[1, 3, 5]
    return newER
end

function update_MPOMPOEnvL(mpoEnvL, mpoTensorK, mpoTensor, mpoTensorB)
    @tensor newEL[-1; -2 -3] := mpoEnvL[1, 3, 5] * mpoTensorK[5, 4, -3, 6] *
                                mpoTensor[3, 2, -2, 4] * conj(mpoTensorB[1, 2, -1, 6])
    return newEL
end

function update_MPOMPOEnvR(mpoEnvR, mpoTensorK, mpoTensor, mpoTensorB)
    @tensor newER[-1 -2; -3] := mpoTensorK[-1, 2, 1, 6] * mpoTensor[-2, 4, 3, 2] *
                                conj(mpoTensorB[-3, 4, 5, 6]) * mpoEnvR[1, 3, 5]
    return newER
end

function initializeMPOEnvironments(finiteMPS::SparseMPS, finiteMPO::SparseMPO;
                                   centerPos::Int64 = 1)

    # get length of finiteMPS
    N = length(finiteMPS)

    # construct MPO environments
    mpoEnvL = Vector{TensorMap{ComplexF64}}(undef, N)
    mpoEnvR = Vector{TensorMap{ComplexF64}}(undef, N)

    # initialize end-points of mpoEnvL and mpoEnvR
    mpoEnvL[1] = ones(ComplexF64, space(finiteMPS[1], 1),
                      space(finiteMPO[1], 1) ⊗ space(finiteMPS[1], 1))
    mpoEnvR[N] = ones(ComplexF64,
                      space(finiteMPS[N], 3)' ⊗ space(finiteMPO[N], 3)',
                      space(finiteMPS[N], 3)')

    # compute mpoEnvL up to (centerPos - 1)
    for siteIdx in 1:+1:(centerPos - 1)
        mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx],
                                              finiteMPO[siteIdx], finiteMPS[siteIdx])
    end

    # compute mpoEnvR up to (centerPos + 1)
    for siteIdx in N:-1:(centerPos + 1)
        mpoEnvR[siteIdx - 1] = update_MPOEnvR(mpoEnvR[siteIdx], finiteMPS[siteIdx],
                                              finiteMPO[siteIdx], finiteMPS[siteIdx])
    end
    return mpoEnvL, mpoEnvR
end

function initializePROEnvironments(ketMPS::SparseMPS, orthStates::Vector{SparseMPS};
                                   centerPos::Int64 = 1)

    # get length of ketMPS
    N = length(ketMPS)

    # construct projector environments
    numOrthStates = length(orthStates)
    projEnvsL = Vector{Vector{TensorMap{ComplexF64}}}(undef, numOrthStates)
    projEnvsR = Vector{Vector{TensorMap{ComplexF64}}}(undef, numOrthStates)
    for orthIdx in eachindex(orthStates)

        # select braMPS
        braMPS = orthStates[orthIdx]

        # initialize projector environments
        projEnvL = Vector{TensorMap{ComplexF64}}(undef, N)
        projEnvR = Vector{TensorMap{ComplexF64}}(undef, N)

        # initialize end-points of projEnvL and projEnvR
        projEnvL[1] = ones(ComplexF64, space(braMPS[1], 1), space(ketMPS[1], 1))
        projEnvR[N] = ones(ComplexF64, space(ketMPS[N], 3)', space(braMPS[N], 3)')

        # compute projEnvL up to (centerPos - 1)
        for siteIdx in 1:+1:(centerPos - 1)
            projEnvL[siteIdx + 1] = update_MPSEnvL(projEnvL[siteIdx], ketMPS[siteIdx],
                                                   braMPS[siteIdx])
        end

        # compute projEnvR up to (centerPos + 1)
        for siteIdx in N:-1:2
            projEnvR[siteIdx - 1] = update_MPSEnvR(projEnvR[siteIdx], ketMPS[siteIdx],
                                                   braMPS[siteIdx])
        end

        # store projector environments
        projEnvsL[orthIdx] = projEnvL
        projEnvsR[orthIdx] = projEnvR
    end
    return projEnvsL, projEnvsR
end
