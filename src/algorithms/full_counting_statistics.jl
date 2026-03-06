function compute_phase_distribution(
        mS::MassiveSchwingerModel,
        finiteMPS::SparseMPS;
        ϕMax::Float64 = 4π,
        λNums::Int64 = 101,
        verbosePrint::Bool = false,
    )

    # get truncationParameters
    truncationParameters = mS.modelParameters.truncationParameters
    bogoliubovR = truncationParameters[:bogoliubovR]

    # get hamiltonianParameters
    hamiltonianParameters = mS.modelParameters.hamiltonianParameters
    θ = hamiltonianParameters[:θ]
    m = hamiltonianParameters[:m]
    M = hamiltonianParameters[:M]
    L = hamiltonianParameters[:L]

    # get momentumModes
    modeOccupations = mS.modeOccupations
    momentumModes = modeOccupations[1, :]

    # get length of finiteMPS
    numSites = length(finiteMPS)
    chainCenter = Int((numSites + 1) / 2)

    # construct kroneckerDeltaMPS
    physSpaces = mS.physSpaces
    kronDelSpaces = [
        removeDegeneracyQN(fuse(physSpace, conj(flip(physSpace))))
            for
            physSpace in physSpaces
    ]
    fullKroneckerDeltaMPS = generateKroneckerDeltaMPS(kronDelSpaces)
    # combinedVecSpace = space(fullKroneckerDeltaMPS[chainCenter], 4)';

    # # construct generalized kroneckerDeltaMPS
    # physSpaces = getPhysicalVectorSpaces(finiteMPS);
    # kronDelSpaces = [removeDegeneracyQN(fuse(physSpace, conj(flip(physSpace)))) for physSpace in physSpaces];
    # fullKroneckerDeltaMPS = generateGeneralKroneckerDeltaMPS(kronDelSpaces);
    # combinedVecSpace = space(fullKroneckerDeltaMPS[chainCenter], 4)';

    # # get ordering of QNs in combinedVecSpace
    # productSectors = combinedVecSpace.dims;
    # combinedVecSpaceOrdering = [productSector.charge for productSector in keys(productSectors)];
    # translationOperator = real.(diagm(exp.(+1im * L / 2 * 2π / L * combinedVecSpaceOrdering)));
    # translationOperator = TensorMap(translationOperator, combinedVecSpace, combinedVecSpace);

    # set lambdaValues
    R = 1.0
    λStep = 2 * ϕMax / (λNums - 1)
    lambdaValues = ϕMax / (0.5 * (λNums - 1)) .* collect((-0.5 * (λNums - 1)):-1)

    # initialize array to store expectation values
    phiFieldExpVals = zeros(Float64, length(lambdaValues))

    # loop over all λ
    for idxL in eachindex(lambdaValues)

        # get λ
        λ = lambdaValues[idxL]
        verbosePrint && @printf("λ step %d/%d\n", idxL, length(lambdaValues))

        # construct local vertex operator V_{λ, 0}(0, 0)
        localOperators = Vector{TensorMap{ComplexF64}}(undef, numSites)
        for (siteIdx, momentumVal) in enumerate(momentumModes)
            physSpace = physSpaces[siteIdx]
            if momentumVal == 0
                localOperators[siteIdx] = localVertexOp_mS(
                    momentumVal, physSpace, λ, M, L,
                    bogoliubovR
                )
            else
                localOperators[siteIdx] = exp(-(λ / R)^2 / (2 * abs(momentumVal))) *
                    localVertexOp_mS(
                    momentumVal, physSpace, λ, M, L,
                    bogoliubovR
                )
            end
        end
        localOperators = SparseMPO(localOperators)

        # construct vertexMPO
        vertexMPO = convertLocalOperatorsToMPO(localOperators, fullKroneckerDeltaMPS)

        # # construct fullVertexMPO
        # fullVertexMPO = convertLocalOperatorsToGeneralMPO(localOperators, fullKroneckerDeltaMPS);

        # # construct phiMPO
        # phiMPO = contractPhiMPO(fullVertexMPO, translationOperator, truncError = 1e-4);

        # compute expectation value of phiMPO operator
        expVal_phiField = expectation_value_mpo(finiteMPS, vertexMPO)
        # expVal_phiField = expectation_value_mpo(finiteMPS, phiMPO);
        if imag(expVal_phiField) > 1.0e-12
            ErrorException("complex expectation value found, check MPO construction.")
        end
        phiFieldExpVals[idxL] = real(expVal_phiField)
    end

    # mirror phiFieldExpVals to have full [-ϕMax, ..., 0, ..., +ϕMax] range
    phiFieldExpVals = vcat(phiFieldExpVals, 1, reverse(phiFieldExpVals))

    # perform discrete Fourier transformation of P(φ) = DFT(F(λ))
    frequencies, fourierComponents = NFT(λNums, λStep, phiFieldExpVals)
    # display(frequencies)
    # display(fourierComponents)

    # function return
    return phiFieldExpVals, frequencies, fourierComponents
end

function NFT(numPoints, pointSpacing, samplingPoints)
    """ computes numerical Fourier transformation of a discrete set of values """

    frequencies = fftshift(fftfreq(numPoints)) ./ pointSpacing
    fourierComponents = fftshift(fft(ifftshift(samplingPoints))) .* pointSpacing
    return frequencies, fourierComponents
end

function NFT2(F, xmax, dx)
    x = collect((-xmax):dx:(+xmax))
    y = F.(x)
    fourierComponents = fftshift(fft(ifftshift(y))) .* dx
    frequencies = fftshift(fftfreq(length(x))) ./ dx
    return frequencies, fourierComponents
end
