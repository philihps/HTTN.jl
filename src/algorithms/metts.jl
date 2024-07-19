""" METTS to compute thermal state properties """

#=

This example code implements the minimally entangled typical thermal state (METTS).
For more information on METTS, see the following references:
- "Minimally entangled typical quantum states at finite temperature", Steven R. White,
  Phys. Rev. Lett. 102, 190601 (2009)
  and arxiv:0902.4475 (https://arxiv.org/abs/0902.4475)
- "Minimally entangled typical thermal state algorithms", E M Stoudenmire and Steven R White,
  New Journal of Physics 12, 055026 (2010) https://doi.org/10.1088/1367-2630/12/5/055026

=#

@kwdef struct METTS2
    numWarmUp::Int64 = 10
    numMETTS::Int64 = 2000
    krylovDim::Int = 2
    compressionAlg::String = "zipUp"
    truncErrT::Float64 = 1e-6
    truncErrK::Float64 = 1e-8
    truncErrM::Float64 = 1e-10
    doBasisExtend::Bool = true
    verbosePrint = false
end

"""
Given a Vector of numbers, returns
the average and the standard error
(= the width of distribution of the numbers)
"""
function avg_err(v::Vector)
  N = length(v)
  avg = sum(v) / N;
  avg2 = sum(v.^2) / N;
  return avg, sqrt(abs((avg2 - avg^2)) / N)
end

function removeQNsMPS(finiteMPS::SparseMPS)

    # remove QN and convert to ComplexSpace
    newFiniteMPS = Vector{TensorMap}(undef, length(finiteMPS));
    for siteIdx = 1 : length(finiteMPS)
        tensorDims = [dim(space(finiteMPS[siteIdx], tensorIdx)) for tensorIdx = 1 : 3];
        newFiniteMPS[siteIdx] = TensorMap(convert(Array, finiteMPS[siteIdx]), ComplexSpace(tensorDims[1]) ⊗ ComplexSpace(tensorDims[2]), ComplexSpace(tensorDims[3]));
    end
    return SparseMPS(newFiniteMPS);

end

function sample_from_MPS(finiteMPS::SparseMPS)
    """ Returns one sample of the probability distribution defined by squaring the components of the tensor that the MPS represents """

    # initialize sampleResult
    sampleResult = zeros(Int64, length(finiteMPS));

    # compute MPS sample for negative momenutum modes
    for siteIdx in eachindex(finiteMPS)

        if siteIdx == 1

            # get physical vector space
            physSpace = space(finiteMPS[siteIdx], 2);
            dimPhysSpace = dim(physSpace);

            # compute the probability of each state one-by-one and stop when the random number r is below the total prob so far
            pDisc = 0.0;
            randNum = rand();
            n = 1;
            An = similar(finiteMPS[siteIdx]);
            pn = 0.0;
            while n <= dimPhysSpace
                projVector = zeros(Float64, dimPhysSpace);
                projVector[n] = 1.0;
                projVector = TensorMap(projVector, one(physSpace), physSpace);
                @tensor An[-1; -2] := projVector[2] * finiteMPS[siteIdx][-1, 2, -2];
                pn = real(tr(An' * An));
                pDisc += pn;
                (randNum < pDisc) && break;
                n += 1;
            end
            sampleResult[siteIdx] = n;

            if siteIdx < length(finiteMPS)
                @tensor A[-1 -2; -3] := An[-1, 1] * finiteMPS[siteIdx + 1][1, -2, -3];
                A *= (1.0 / sqrt(pn));
            end

        elseif mod(siteIdx, 2) == 0

            # get physical vector spaces
            physSpaceL = space(finiteMPS[siteIdx + 0], 2);
            physSpaceR = space(finiteMPS[siteIdx + 1], 2);
            dimPhysSpaceL = dim(physSpaceL);
            dimPhysSpaceR = dim(physSpaceR);
            @assert dimPhysSpaceL == dimPhysSpaceR

            # compute the probability of each state one-by-one and stop when the random number r is below the total prob so far
            pDisc = 0.0;
            randNum = rand();
            n = 1;
            An = similar(finiteMPS[siteIdx]);
            pn = 0.0;
            while n <= dimPhysSpaceL && n <= dimPhysSpaceR
                projVectorL = zeros(Float64, dimPhysSpaceL);
                projVectorR = zeros(Float64, dimPhysSpaceR);
                projVectorL[n] = 1.0;
                projVectorR[n] = 1.0;
                projVector = TensorMap(projVectorL * projVectorR', one(physSpaceL) ⊗ one(physSpaceR), physSpaceL ⊗ physSpaceR);
                @tensor An[-1; -2] := projVector[2, 3] * finiteMPS[siteIdx + 0][-1, 2, 1] * finiteMPS[siteIdx + 1][1, 3, -2];
                pn = real(tr(An' * An));
                pDisc += pn;
                (randNum < pDisc) && break;
                n += 1;
            end
            sampleResult[siteIdx + 0] = n;
            sampleResult[siteIdx + 1] = n;

            if siteIdx < length(finiteMPS) - 1
                @tensor A[-1 -2; -3] := An[-1, 1] * finiteMPS[siteIdx + 2][1, -2, -3];
                A *= (1.0 / sqrt(pn));
                finiteMPS[siteIdx + 2] = A;
            end

        end

    end
    return sampleResult;

end

# function sample_from_MPS(finiteMPS::SparseMPS)
#     """ Returns one sample of the probability distribution defined by squaring the components of the tensor that the MPS represents """

#     # remove QNs from MPS
#     finiteMPS = removeQNsMPS(finiteMPS);

#     # initialize sampleResult
#     sampleResult = zeros(Int64, length(finiteMPS));

#     # compute MPS sample for negative momenutum modes
#     A = finiteMPS[1];
#     for siteIdx = 1 : +1 : length(finiteMPS)

#         # get physical vector space
#         physVecSpace = space(finiteMPS[siteIdx], 2);
#         dimPhysVecSpace = dim(physVecSpace);

#         # compute the probability of each state one-by-one and stop when the random number r is below the total prob so far
#         pDisc = 0.0;
#         randNum = rand();
#         n = 1;
#         An = similar(A);
#         pn = 0.0;
#         while n <= dimPhysVecSpace
#             projVector = zeros(Float64, dimPhysVecSpace);
#             projVector[n] = 1.0;
#             projVector = TensorMap(projVector, one(physVecSpace), physVecSpace);
#             @tensor An[-1; -2] := projVector[2] * A[-1, 2, -2];
#             pn = real(tr(An' * An));
#             pDisc += pn;
#             (randNum < pDisc) && break;
#             n += 1;
#         end
#         sampleResult[siteIdx] = n;

#         if siteIdx < length(finiteMPS)
#             @tensor A[-1 -2; -3] := An[-1, 1] * finiteMPS[siteIdx + 1][1, -2, -3];
#             A *= (1.0 / sqrt(pn));
#         end

#     end
#     return sampleResult;

# end

function metts(finiteMPS::SparseMPS, finiteMPO::SparseMPO, timeStep::Union{Float64, ComplexF64}, finalBeta::Union{Int64, Float64}, alg::METTS2)

    # make timeRanges and check timeStep is commensurate
    timeRanges = timeStep : timeStep : (finalBeta / 2)
    if norm(length(timeRanges) * timeStep - finalBeta / 2) > 1e-10
    error("Time step timeStep=$timeStep not commensurate with finalBeta/2=$(finalBeta/2)")
    end

    energies = zeros(Float64, 0, 2);

    # main METTS loop
    for step in 1 : (alg.numWarmUp + alg.numMETTS)

        if step <= alg.numWarmUp
            println("making warmup METTS number $step")
        else
            println("making actual METTS number $(step - alg.numWarmUp)")
        end

        #-----------------------------------------------------------------
        # perform time step by applying exp(-timeStep * H)

        # make basis extension to include a number of global Krylov vectors
        if alg.doBasisExtend
            finiteMPS = basisExtend(finiteMPS, finiteMPO, TDVP2());
        end

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO);

        # perform imaginary time evolution up to finalBeta
        for timeIdx in eachindex(timeRanges)

            # initialize truncationErrors
            truncationErrors = Float64[];

            # sweep L ---> R
            for siteIdx = 1 : +1 : (length(finiteMPS) - 1)

                # construct initial AC
                AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
                newAC2, convHist = exponentiate(x -> applyAC2(x, finiteMPO[siteIdx + 0], finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 1]), -1im * timeStep / 2, AC2, Lanczos());
                # newAC2, convHist = exponentiate(x -> applyAC2(x, finiteMPO[siteIdx + 0], finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 1]), -1 * timeStep / 2, AC2, Lanczos());

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newAC2, (1, 2), (3, 4), trunc = truncerr(alg.truncErrT), alg = TensorKit.SVD());
                S /= norm(S);
                U = permute(U, (1, 2), (3, ));
                V = permute(S * V, (1, 2), (3, ));
                truncationErrors = vcat(truncationErrors, ϵ);

                # assign updated tensors
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

                # compute K(n + 1) and apply it to V(n + 1) to evolve it with exp(+1im * timeStep/2 * K(n + 1))
                if siteIdx < (length(finiteMPS) - 1)
                    # H_AC = ∂∂AC(siteIdx + 1, finiteMPO, mpoEnvL, mpoEnvR);
                    # finiteMPS[siteIdx + 1], convHist = exponentiate(H_AC, +1im * timeStep / 2, finiteMPS[siteIdx + 1], Lanczos());
                    # finiteMPS[siteIdx + 1], convHist = exponentiate(H_AC, +1 * timeStep / 2, finiteMPS[siteIdx + 1], Lanczos());
                    # finiteMPS[siteIdx + 1], convHist = exponentiate(x -> applyAC(x, finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 1], mpoEnvR[siteIdx + 1]), +1im * timeStep / 2, finiteMPS[siteIdx + 1], Lanczos());
                    finiteMPS[siteIdx + 1], convHist = exponentiate(x -> applyAC(x, finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 1], mpoEnvR[siteIdx + 1]), +1 * timeStep / 2, finiteMPS[siteIdx + 1], Lanczos());
                end

            end

            # sweep L <--- R
            for siteIdx = (length(finiteMPS) - 1) : -1 : 1

                # construct initial AC
                AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
                # newAC2, convHist = exponentiate(x -> applyAC2(x, finiteMPO[siteIdx + 0], finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 1]), -1im * timeStep / 2, AC2, Lanczos());
                newAC2, convHist = exponentiate(x -> applyAC2(x, finiteMPO[siteIdx + 0], finiteMPO[siteIdx + 1], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 1]), -1 * timeStep / 2, AC2, Lanczos());

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newAC2, (1, 2), (3, 4), trunc = truncerr(alg.truncErrT), alg = TensorKit.SVD());
                S /= norm(S);
                U = permute(U * S, (1, 2), (3, ));
                V = permute(V, (1, 2), (3, ));
                truncationErrors = vcat(truncationErrors, ϵ);

                # assign updated tensors
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1], finiteMPS[siteIdx + 1], finiteMPO[siteIdx + 1], finiteMPS[siteIdx + 1]);

                # compute K(n) and apply it to V(n) to evolve it with exp(+1im * timeStep/2 * K(n))
                if siteIdx > 1
                    # H_AC = ∂∂AC(siteIdx + 0, finiteMPO, mpoEnvL, mpoEnvR);
                    # finiteMPS[siteIdx + 0], convHist = exponentiate(H_AC, +1im * timeStep / 2, finiteMPS[siteIdx + 0], Lanczos());
                    # finiteMPS[siteIdx + 0], convHist = exponentiate(H_AC, +1 * timeStep / 2, finiteMPS[siteIdx + 0], Lanczos());
                    # finiteMPS[siteIdx + 0], convHist = exponentiate(x -> applyAC(x, finiteMPO[siteIdx + 0], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 0]), +1im * timeStep / 2, finiteMPS[siteIdx + 0], Lanczos());
                    finiteMPS[siteIdx + 0], convHist = exponentiate(x -> applyAC(x, finiteMPO[siteIdx + 0], mpoEnvL[siteIdx + 0], mpoEnvR[siteIdx + 0]), +1 * timeStep / 2, finiteMPS[siteIdx + 0], Lanczos());
                end

            end

        end

        # # get MPS linkDims
        # println(getLinkDimsMPS(finiteMPS))

        #-----------------------------------------------------------------

        # measure properties after >= alg.numWarmUp METTS have been made
        if step > alg.numWarmUp
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO);
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal);
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            a_E, err_E = avg_err(energies[:, 1]);
            energies = vcat(energies, [mpoExpVal a_E]);
            @printf("energy of METTS at step %d = %0.4f\n", step - alg.numWarmUp, mpoExpVal)
            @printf(
                "estimated Energy = %0.4f ± %0.4f  /  [%0.4f, %0.4f]\n",
                a_E,
                err_E,
                a_E - err_E,
                a_E + err_E
            )
        end

        # # initialize random basis vector with mixing between opposite momentum modes
        # initialTensors = Vector{TensorMap}(undef, length(finiteMPS));
        # for siteIdx = eachindex(finiteMPS)
        #     if siteIdx == 1
        #         physSpace = space(finiteMPS[siteIdx], 2);
        #         initialTensor = rand(Float64, 1, dim(physSpace), 1);
        #         initialTensors[siteIdx] = TensorMap(initialTensor, U1Space(0 => 1) ⊗ physSpace, U1Space(0 => 1));
        #     elseif mod(siteIdx, 2) == 0
        #         physSpaceL = space(finiteMPS[siteIdx + 0], 2);
        #         physSpaceR = space(finiteMPS[siteIdx + 1], 2);
        #         productSpace = fuse(U1Space(0 => 1), physSpaceL);
        #         initialTensorL = zeros(Float64, 1, dim(physSpaceL), dim(productSpace));
        #         initialTensorR = zeros(Float64, dim(productSpace), dim(physSpaceR), 1);
        #         randomOccupations = rand(dim(productSpace));
        #         for (idxN, occP) in enumerate(randomOccupations)
        #             initialTensorL[1, idxN, idxN] = occP;
        #             initialTensorR[idxN, idxN, 1] = occP;
        #         end
        #         initialTensors[siteIdx + 0] = TensorMap(initialTensorL, U1Space(0 => 1) ⊗ physSpaceL, productSpace);
        #         initialTensors[siteIdx + 1] = TensorMap(initialTensorR, productSpace ⊗ physSpaceR, U1Space(0 => 1));
        #     end
        # end
        # randomMPS = SparseMPS(initialTensors, normalizeMPS = true);

        # # collapse time evolved MPS and compute probability
        # transitionProbability = abs(dotMPS(randomMPS, finiteMPS));
        # # @printf("transition probability = %0.4f\n", transitionProbability)

        # # update finiteMPS
        # finiteMPS = 1 / sqrt(transitionProbability) * randomMPS;

        # sample in harmonic oscillator basis
        mpsSample = sample_from_MPS(finiteMPS);
        println(mpsSample)

        # create new block product state from mpsSample
        initialTensors = Vector{TensorMap}(undef, length(finiteMPS));
        for siteIdx = eachindex(finiteMPS)
            if siteIdx == 1
                physSpace = space(finiteMPS[siteIdx], 2);
                initialTensor = zeros(Float64, 1, dim(physSpace), 1);
                initialTensor[mpsSample[siteIdx]] = 1.0;
                initialTensors[siteIdx] = TensorMap(initialTensor, U1Space(0 => 1) ⊗ physSpace, U1Space(0 => 1));
            elseif mod(siteIdx, 2) == 0
                physSpaceL = space(finiteMPS[siteIdx + 0], 2);
                physSpaceR = space(finiteMPS[siteIdx + 1], 2);
                productSpace = fuse(U1Space(0 => 1), physSpaceL);
                initialTensorL = zeros(Float64, 1, dim(physSpaceL), dim(productSpace));
                initialTensorR = zeros(Float64, dim(productSpace), dim(physSpaceR), 1);                
                idxNL = mpsSample[siteIdx + 0];
                idxNR = mpsSample[siteIdx + 1];
                initialTensorL[1, idxNL, idxNL] = 1.0;
                initialTensorR[idxNR, idxNR, 1] = 1.0;
                initialTensors[siteIdx + 0] = TensorMap(initialTensorL, U1Space(0 => 1) ⊗ physSpaceL, productSpace);
                initialTensors[siteIdx + 1] = TensorMap(initialTensorR, productSpace ⊗ physSpaceR, U1Space(0 => 1));
            end
        end
        finiteMPS = SparseMPS(initialTensors, normalizeMPS = true);

        # # create new block product state from mpsSample
        # initialTensors = Vector{TensorMap}(undef, length(finiteMPS));
        # for siteIdx = eachindex(finiteMPS)
        #     physSpace = space(finiteMPS[siteIdx], 2);
        #     initTensor = zeros(Float64, 1, dim(physSpace), 1);
        #     initTensor[1, mpsSample[siteIdx], 1] = 1.0;
        #     initialTensors[siteIdx] = TensorMap(initTensor, U1Space(0 => 1) ⊗ physSpace, U1Space(0 => 1));
        # end
        # finiteMPS = SparseMPS(initialTensors);

        # # Measure in X or Z basis on alternating steps
        # if step % 2 == 1
        #     psi = apply(Ry_gates, psi)
        #     samp = sample_from_MPS(psi)
        #     new_state = [samp[j] == 1 ? "X+" : "X-" for j in 1:N]
        # else
        #     samp = sample_from_MPS(psi)
        #     new_state = [samp[j] == 1 ? "Z+" : "Z-" for j in 1:N]
        # end
        # psi = MPS(s, new_state)
        # end

    end

    return energies

end