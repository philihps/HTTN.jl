""" METTS to compute thermal state properties """
#=
This example code implements the minimally entangled typical thermal state (METTS).
For more information on METTS, see the following references:
"Minimally entangled typical quantum states at finite temperature", Steven R. White,
  Phys. Rev. Lett. 102, 190601 (2009)
  and arxiv:0902.4475 (https://arxiv.org/abs/0902.4475)
"Minimally entangled typical thermal state algorithms", E M Stoudenmire and Steven R White,
  New Journal of Physics 12, 055026 (2010) https://doi.org/10.1088/1367-2630/12/5/055026
=#
@kwdef struct METTS2
    numWarmUp::Int64 = 10
    numMETTS::Int64 = 500
    krylovDim::Int = 2
    bondDim::Int = 1000
    compressionAlg::String = "zipUp"
    truncErrT::Float64 = 1e-6 # after applying 2-body gate in TVDP
    truncErrK::Float64 = 1e-8 # from basis extension
    truncErrM::Float64 = 1e-10
    tol::Float64 = 1.0
    doBasisExtend::Bool = true
    verbosePrint = false
end
"""
Given a Vector of numbers, returns
the average and the standard error
(= the width of distribution of the numbers)
"""
function avg_stderr(v::Vector)
    N = length(v)
    avg = sum(v) / N
    avg2 = sum(v .^ 2) / N
    return avg, sqrt(abs((avg2 - avg^2)) / N)
end

function sample_ZM(localTensor, physSpace)
    """
    Sample zero mode

    Returns
    - n: index of sampled basis state
    - pn: probability of localTensor to be in the n-th basis state
    - An: collapsed localTensor
    """
    pDisc = 0.0
    randNum = rand()
    An = similar(localTensor)
    n = 1
    pn = 0.0

    while n <= dim(physSpace)
        # create projectors defined by basis state
        targetQN = U1Space(0 => 1)
        projVector = zeros(Float64, 1, dim(physSpace))
        projVector[1, n] = 1.0
        projVector = TensorMap(projVector, targetQN, physSpace)
       
        # collapse basis state onto localTensor
        @tensor An[-1 -2; -3] := projVector[-2, 2] * localTensor[-1, 2, -3]
        pn = real(tr(An' * An))
        pDisc += real(pn)
        (randNum < pDisc) && break
        n += 1
    end
    return n, pn, An
end

function sample_non_ZM(localTensor, physSpace)
    pDisc = 0.0
    randNum = rand()
    n = 1
    pn = 0.0
    An = similar(localTensor)

    dimPhysSpace = dim(physSpace)
    physSpaceQNs = physSpace.dims.keys

    while n <= dimPhysSpace
        targetQN = U1Space(physSpaceQNs[n] => 1)
        projVector = TensorMap(ones, targetQN, physSpace)

        @tensor An[-1 -2; -3] := projVector[-2, 2] * localTensor[-1, 2, -3]
        pn = real(tr(An' * An))
        pDisc += pn
        (randNum < pDisc) && break
        n += 1
    end
    return n, pn, An
end

function sample_basis(localTensor, localProj, index::Int64, physSpace, phyVecSpaceOrdering)
    """
    Sample local tensor given projection operators

    Returns
    - n: index of sampled basis state
    - pn: probability of localTensor to be in the n-th basis state
    - An: collapsed localTensor
    """
    pDisc = 0.0
    randNum = rand()
    An = similar(localTensor)
    n = 1
    pn = 0.0
    dimPhysSpace = dim(physSpace)
    while n <= dimPhysSpace
        # create projectors defined by basis state
        if index == 1
            targetQN = U1Space(phyVecSpaceOrdering[1] => 1)
            projVector = zeros(Float64, 1, dimPhysSpace)
            projVector[1, n] = 1.0
            projVector = TensorMap(projVector, targetQN, physSpace)
        else
            targetQN = U1Space(phyVecSpaceOrdering[n] => 1)
            proj = blocks(localProj)[U1Irrep(phyVecSpaceOrdering[n])][1]

            projVector = proj * TensorMap(ones, targetQN, physSpace)
        end
        # collapse basis state onto localTensor
        @tensor An[-1 -2; -3] := projVector[-2, 2] * localTensor[-1, 2, -3]
        pn = real(tr(An' * An))
        pDisc += pn
        (randNum < pDisc) && break
        n += 1
    end
    return n, pn, An
end

function sample_pair_basis_1(localTensor, modeProj, qNsL, qNsR)
    """
    Sample mode tensor using projection operators in sqOp basis

    Params:
    - localTensor: 3-leg tensor for siteIdx = 1, 4-leg tensor for siteIdx % 2 = 0

    Returns:
    - n: index tuple for sampled pair mode (-k, k'+)
    """
    pDisc = 0.0
    randNum = rand()
    An = similar(localTensor)
    n = (0,0)
    pn = 0.0

    codoms, doms = map(fusiontrees(modeProj)) do (f1, f2)
        (f1, f2)
    end |> x -> (map(first, x), map(last, x))

    # define projector for each possible pair mode in the codomain
    for codom in unique(codoms)
        singleModeProj = complex(TensorMap(zeros, codomain(modeProj), domain(modeProj)))

        # ensure indices matching
        @assert (fusiontrees(singleModeProj)).rowr == (fusiontrees(modeProj)).rowr
        @assert (fusiontrees(singleModeProj)).colr == (fusiontrees(modeProj)).colr

        indicesCodom = findall(x -> x == codom, codoms)
        # find the corresponding pair mode in the domain
        for i in indicesCodom
            singleModeProj[codoms[i], doms[i]] = modeProj[codoms[i], doms[i]]
        end

        qNs = codom.uncoupled
        n = (indexin(qNs[1].charge, [qN.charge for qN in qNsL])[1], indexin(qNs[2].charge, [qN.charge for qN in qNsR])[1])
        @tensor An[-1 -2 -3; -4] := singleModeProj[-2, -3, 1, 2] * localTensor[-1, 1, 2, -4]
        pn = real(tr(An' * An))

        pDisc += pn
        (randNum < pDisc) && break
    end

    return n, pn, An
end

function sample_pair_basis_2(localTensor, eigState, qNsL, qNsR)
    """
    Sample mode tensor using projection operators in Fock basis

    Params:
    - localTensor: 3-leg tensor for siteIdx = 1, 4-leg tensor for siteIdx % 2 = 0

    Returns:
    - n: index tuple for sampled pair mode (-k, k'+)
    """
    pDisc = 0.0
    randNum = rand()
    An = similar(localTensor)
    n = (0,0)
    pn = 0.0

    codoms, doms = map(fusiontrees(eigState)) do (f1, f2)
        (f1, f2)
    end |> x -> (map(first, x), map(last, x))

    # define projector for each possible pair mode in the codomain
    for codom in unique(codoms)
        codomL, codomR = codom.uncoupled
        pairModeProj = zeros(ComplexF64, 1, 1, length(qNsL), length(qNsR))
        indicesCodom = findall(x -> x == codom, codoms)

        # find the corresponding pair mode in the domain
        for i in indicesCodom
            indexL, indexR = doms[i].uncoupled
            indexL, indexR = indexL.charge, indexR.charge
            @assert indexL + indexR == codomL.charge + codomR.charge # check momentum preservation
            posL, posR = indexin(indexL, [qN.charge for qN in qNsL])[1], indexin(indexR, [qN.charge for qN in qNsR])[1]
            pairModeProj[1, 1, posL, posR] = 1.0
        end

        pairModeProj = TensorMap(pairModeProj, U1Space(codomL.charge => 1) ⊗ U1Space(codomR.charge => 1), domain(eigState))

        qNs = codom.uncoupled
        n = (indexin(qNs[1].charge, [qN.charge for qN in qNsL])[1], indexin(qNs[2].charge, [qN.charge for qN in qNsR])[1])
        @tensor An[-1 -2 -3; -4] := pairModeProj[-2, -3, 1, 2] * localTensor[-1, 1, 2, -4]
        pn = real(tr(An' * An))

        pDisc += pn
        (randNum < pDisc) && break
    end

    return n, pn, An
end

function sample_to_CPS(mpsSample, momSample, finiteMPS)
    virtSpaces = vcat(U1Space(0 => 1),
                      [U1Space(sum(momSample[1:linkIdx]) => 1)
                       for linkIdx in 1:(length(momSample))])
    # create new classical product state from mpsSample
    initialTensors = Vector{TensorMap}(undef, length(finiteMPS))
    for siteIdx in eachindex(finiteMPS)
        physSpace = space(finiteMPS[siteIdx], 2)
        virtSpaceL = virtSpaces[siteIdx + 0]
        virtSpaceR = virtSpaces[siteIdx + 1]
        initTensor = zeros(Float64, dim(virtSpaceL), dim(physSpace), dim(virtSpaceR))
        initTensor[1, mpsSample[siteIdx], 1] = 1.0
        initialTensors[siteIdx] = TensorMap(initTensor, virtSpaceL ⊗ physSpace, virtSpaceR)
    end
    return SparseMPS(initialTensors)
end

function sample_to_BPS(mpsSample, momSample, finiteMPS, coeffs, modeSectors)
    virtSpaces = vcat(U1Space(0 => 1),
                      [U1Space(sum(momSample[1:linkIdx + 1]) => 1)
                       for linkIdx in 1:(length(momSample)) if linkIdx%2==0])
    blockState = Vector{TensorMap}(undef, length(virtSpaces))
    i = 1
    
    for siteIdx in eachindex(finiteMPS)
        if siteIdx == 1
            physSpace = space(finiteMPS[siteIdx], 2)
            virtSpaceL = U1Space(0 => 1)
            virtSpaceR = U1Space(0 => 1)
            initTensor = zeros(ComplexF64, dim(virtSpaceL), dim(physSpace), dim(virtSpaceR))
            initTensor[1, mpsSample[siteIdx], 1] = 1.0
            blockState[i] = TensorMap(initTensor, virtSpaceL ⊗ physSpace, virtSpaceR)
            i += 1
        
        elseif siteIdx % 2 == 0
            virtSpaceL = virtSpaces[siteIdx ÷ 2]
            virtSpaceR = virtSpaces[siteIdx ÷ 2 + 1]
            physSpaceL = space(finiteMPS[siteIdx], 2)
            physSpaceR = space(finiteMPS[siteIdx + 1], 2)
            qNsL = [qN.charge for qN in physSpaceL.dims.keys]
            qNsR = [qN.charge for qN in physSpaceR.dims.keys]

            initTensor = zeros(ComplexF64, 1, dim(physSpaceL), dim(physSpaceR), 1)
            
            totalMom = momSample[siteIdx + 0] + momSample[siteIdx + 1]
            coeff = coeffs[siteIdx ÷ 2].data[U1Irrep(totalMom)]
            fusionTree = modeSectors[siteIdx ÷ 2][U1Irrep(totalMom)]
            for (pairMode, posTK) in fusionTree
                indexL, indexR = pairMode.uncoupled
                indexL, indexR = indexL.charge, indexR.charge
                posL, posR = indexin(indexL, qNsL)[1], indexin(indexR, qNsR)[1]
                initTensor[1, posL, posR, 1] = coeff[posTK[1], posTK[1]]
            end
            blockState[i] = TensorMap(initTensor, virtSpaceL ⊗ physSpaceL ⊗ physSpaceR, virtSpaceR)

            i += 1
        end
    end
            
    return blockState
end

function sample_MPS!(finiteMPS::SparseMPS)
    """
    Returns one sample of the probability distribution defined by squaring the components of the tensor that the MPS represents
    Returns:
    - sampleResult: vector contains index of basis state for each site
    - sampleMomentum: vector contains momentum mode   
    """
    sampleResult = zeros(Int64, length(finiteMPS))
    sampleMomentum = zeros(Int64, length(finiteMPS))
    # sample from probability distribution given by finiteMPS
    for siteIdx in eachindex(finiteMPS)
       
        physSpace = space(finiteMPS[siteIdx], 2)

        if siteIdx == 1
            n, pn, An = sample_ZM(finiteMPS[1], physSpace)
            sampleResult[siteIdx] = n
            sampleMomentum[siteIdx] = 0
        else
            physSpaceQNs = physSpace.dims.keys
            n, pn, An = sample_non_ZM(finiteMPS[siteIdx],physSpace)
            sampleResult[siteIdx] = n
            sampleMomentum[siteIdx] = physSpaceQNs[n].charge
        end
        # fuse left virtual index and (fixed) physical index to transfer information about sample outcome on one site to the next site
        fusionIsometry = isometry(fuse(space(An, 1), space(An, 2)),
                                  space(An, 1) ⊗ space(An, 2))

        if siteIdx < length(finiteMPS)
            @tensor A[-1 -2; -3] := fusionIsometry[-1, 1, 2] * An[1, 2, 3] *
                                    finiteMPS[siteIdx + 1][3, -2, -3]
            A *= (1.0 / sqrt(pn))
            finiteMPS[siteIdx + 1] = A
        end
    end
    return sampleResult, sampleMomentum
end


function sample_MPS_pair!(finiteMPS::SparseMPS, eigStates)
    """
    Returns one sample of the probability distribution defined by squaring the components of the tensor that the MPS represents
    Returns:
    - sampleResult: vector contains index of basis state for each site
    - sampleMomentum: vector contains momentum mode
    """
    sampleResult = zeros(Int64, length(finiteMPS))
    sampleMomentum = zeros(Int64, length(finiteMPS))

    for siteIdx in eachindex(finiteMPS)
        if siteIdx == 1
            n, pn, An = sample_ZM(finiteMPS[1], space(finiteMPS[1], 2))
            sampleResult[siteIdx] = n
            sampleMomentum[siteIdx] = 0

            fusionIsometry = isometry(fuse(space(An, 1), space(An, 2)),
                                  space(An, 1) ⊗ space(An, 2))
            @tensor A[-1 -2; -3] := fusionIsometry[-1, 1, 2] * An[1, 2, 3] *
                                    finiteMPS[siteIdx + 1][3, -2, -3]
            A *= (1.0 / sqrt(pn))
            finiteMPS[siteIdx + 1] = A

        elseif siteIdx % 2 == 0
            qNsL, qNsR =  space(finiteMPS[siteIdx + 0], 2).dims.keys, space(finiteMPS[siteIdx + 1], 2).dims.keys
            @tensor localTensor[-1 -2 -3; -4] := finiteMPS[siteIdx + 0][-1, -2, 1] * finiteMPS[siteIdx + 1][1, -3, -4]
            n, pn, An = sample_pair_basis_2(localTensor, eigStates[siteIdx ÷ 2], qNsL, qNsR)
            sampleResult[siteIdx + 0], sampleResult[siteIdx + 1] = n
            sampleMomentum[siteIdx + 0], sampleMomentum[siteIdx + 1] = qNsL[n[1]].charge, qNsR[n[2]].charge

            if siteIdx < length(finiteMPS) - 2 
                fusionIsometry = isometry(fuse(space(An, 1), space(An, 2), space(An, 3)), space(An, 1) ⊗ space(An, 2) ⊗ space(An, 3))
                @tensor A[-1 -2 ; -3] := fusionIsometry[-1, 1, 2, 3] * An[1, 2, 3, 4] * finiteMPS[siteIdx + 2][4,-2, -3]

                A *= (1.0 /sqrt(pn))
                finiteMPS[siteIdx + 2] = A
            end
        end
    end
    
    return sampleResult, sampleMomentum
end

function metts!(finiteMPS::SparseMPS,
               finiteMPO::SparseMPO,
               numTimeStep::Int64,
               finalBeta::Union{Int64,Float64},
               alg::METTS2)
    """
    Returns:
    - energies: energies[:, 1] -> energy at time step i
                energies[:, 2] -> average energy up to time step i
                energies[:, 3] -> standard error up to time step i
    """

    timeRanges = range(0; stop = finalBeta / 2, length = numTimeStep + 1)
    timeStep = 1im * (timeRanges[2] - timeRanges[1])
    println("Running METTS algorithm for timestep: $(timeStep)")

    numMETTSMax = 3000 # hard limit of METTS iterations
    numMETTS = max(alg.numMETTS, numMETTSMax)
    energies = zeros(Float64, 0, 3)
    truncErrs = zeros(Float64, alg.numWarmUp + numMETTS)

    # main METTS loop
    for step in 1:(alg.numWarmUp + numMETTS)
        if step <= alg.numWarmUp
            println("Making warmup METTS number $step")
        else
            println("Making actual METTS number $(step - alg.numWarmUp)")
        end
        # perform time step by applying exp(-timeStep * H)
        for _ in eachindex(timeRanges)
            finiteMPS, _, _, truncErr = perform_timestep!(finiteMPS, finiteMPO, timeStep,
                                                          TDVP2())
            truncErrs[step] = truncErr
        end
        # measure properties after >= alg.numWarmUp METTS have been made
        if step > alg.numWarmUp
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO)
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal)
            else
                ErrorException("The Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            av_E, err_E = avg_stderr(energies[:, 1])
            energies = vcat(energies, [mpoExpVal av_E err_E])
            @printf("Energy of METTS at step %d = %0.4f\n", step - alg.numWarmUp, mpoExpVal)
            @printf("Estimated energy = %0.6f ± %0.6f  /  [%0.6f, %0.6f]\n",
                    av_E,
                    err_E,
                    av_E - err_E,
                    av_E + err_E)
            if err_E > 0 && abs(err_E / av_E) * 100 <= alg.tol
                println("Standard error is within $(alg.tol)% of the average observable at METTS number $(step - alg.numWarmUp)")
                break
            end
        end
        # collapse to a new state with local basis defined by mpsSample and momSample
        mpsSample, momSample = sample_MPS!(finiteMPS)
        println("Sample of local basis (index): $(mpsSample)")
        println("Sample of local basis (momentum): $(momSample)")
        finiteMPS = sample_to_CPS(mpsSample, momSample, finiteMPS)
    end

    _, av_E_last, err_E_last = energies[end, :]
    if abs(err_E_last / av_E_last) * 100 > alg.tol
        println("The observable does not converge within $numMETTS iterations.")
    end
    return energies, truncErrs
end

function metts(finiteMPS::SparseMPS,
    finiteMPO::SparseMPO,
    numTimeStep::Int64,
    finalBeta::Union{Int64,Float64},
    alg::METTS2)


    return metts!(copy(finiteMPS), finiteMPO::SparseMPO,
    numTimeStep::Int64,
    finalBeta::Union{Int64,Float64},
    alg::METTS2)
end