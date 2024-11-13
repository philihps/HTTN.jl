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

function sample(localMPS, index::Int64, physSpace, phyVecSpaceOrdering)
    """
    Returns
    - n: index of sampled basis state
    - pn: probability of localMPS to be in the n-th basis state
    - An: collapsed localMPS
    """
    pDisc = 0.0
    randNum = rand()
    n = 1
    An = similar(localMPS)
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
            projVector = TensorMap(ones, targetQN, physSpace)
        end
        # collapse basis state onto localMPS
        @tensor An[-1 -2; -3] := projVector[-2, 2] * localMPS[-1, 2, -3]
        pn = real(tr(An' * An))
        pDisc += pn
        (randNum < pDisc) && break
        n += 1
    end
    return n, pn, An
end

function sample_to_CPS(mpsSample, momSample, finiteMPS)
    virtSpaces = vcat(U1Space(0 => 1),
                      [U1Space(sum(momSample[1:linkIdx]) => 1)
                       for linkIdx in 1:(length(momSample) - 1)], U1Space(0 => 1))
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
function sample_from_MPS!(finiteMPS::SparseMPS)
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
        # get physical vector space
        physSpace = space(finiteMPS[siteIdx], 2)
        # get ordering of QNs in physSpace
        physSpaceQNs = physSpace.dims
        phyVecSpaceOrdering = [productSector.charge for productSector in keys(physSpaceQNs)]
        if siteIdx == 1
            n, pn, An = sample(finiteMPS[siteIdx], siteIdx, physSpace, phyVecSpaceOrdering)
            sampleResult[siteIdx] = n
            sampleMomentum[siteIdx] = phyVecSpaceOrdering[1]
        else
            n, pn, An = sample(finiteMPS[siteIdx], siteIdx, physSpace, phyVecSpaceOrdering)
            sampleResult[siteIdx] = n
            sampleMomentum[siteIdx] = phyVecSpaceOrdering[n]
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

function metts(finiteMPS::SparseMPS,
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

    numMETTSMax = 2000 # hard limit of METTS iterations
    numMETTSMin = 10
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
            if step > (alg.numWarmUp + numMETTSMin) && abs(err_E / av_E) * 100 <= alg.tol
                println("Standard error is within $(alg.tol)% of the average observable at METTS number $(step - alg.numWarmUp)")
                break
            end
        end
        # collapse to a new state with local basis defined by mpsSample and momSample
        mpsSample, momSample = sample_from_MPS!(finiteMPS)
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
