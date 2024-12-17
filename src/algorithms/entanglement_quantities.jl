function LinearAlgebra.diag(T::AbstractTensorMap)
    """ Overloading of LinearAlgebra function diag for TensorMap type """

    diagElements = Vector{eltype(T)}()
    blockSectors = blocksectors(T)
    for blockIdx in blockSectors
        append!(diagElements, diag(block(T, blockIdx)))
    end
    return diagElements
end

function compute_entanglement_spectra(finiteMPS::SparseMPS; svCutOff::Float64 = 1e-12)
    """ Returns a vector of entanglement spectra for successive bipartitions of the MPS """

    # initialize list of entanglement spectra
    entanglementSpectra = Vector{Vector{Float64}}(undef, length(finiteMPS) - 1)

    # compute entanglement spectra by two-site SVDs
    for siteIdx in 1:(length(finiteMPS) - 1)

        # orthogonalize MPS
        orthogonalizeMPS!(finiteMPS, siteIdx)

        # construct initial theta
        theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1,), (2, 3)),
                        (1, 2), (3, 4))

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(theta, (1, 2), (3, 4); trunc = truncbelow(svCutOff))
        S /= norm(S)
        entanglementSpectra[siteIdx] = real.(diag(S))
    end
    return entanglementSpectra
end

function compute_entanglement_entropies(finiteMPS::SparseMPS; svCutOff::Float64 = 1e-12)
    """ Compute von-Neumann entropy for a each bipartition along the MPS """

    entanglementSpectra = compute_entanglement_spectra(finiteMPS; svCutOff = svCutOff)
    entEntropy = abs.([-sum(entSpec .^ 2 .* log.(entSpec .^ 2))
                       for entSpec in entanglementSpectra])
    return entEntropy
end

function compute_arbitrary_bipartition(finiteMPS::SparseMPS, openPositions::Vector{Int64})
    """ Computes reduced density matrix for the physical indices that are specified in openPositions """

    # get length of finiteMPS
    N = length(finiteMPS)

    # initialize reduced density matrix
    reducedDensityMatrix = TensorMap(ones, space(finiteMPS[1], 1), space(finiteMPS[1], 1))
    numberOfOpenPhysicalIndices = 0
    for siteIdx in 1:N
        if siteIdx < N
            if any(openPositions .== siteIdx)
                indsRDM = vcat(-collect(1:numberOfOpenPhysicalIndices), 1, 2)
                indsB = [1, -(numberOfOpenPhysicalIndices + 2),
                         -(numberOfOpenPhysicalIndices + 3)]
                indsK = [2, -(numberOfOpenPhysicalIndices + 1),
                         -(numberOfOpenPhysicalIndices + 4)]
                reducedDensityMatrix = ncon([reducedDensityMatrix, finiteMPS[siteIdx],
                                             finiteMPS[siteIdx]], [indsRDM, indsB, indsK],
                                            [false, true, false])
                numberOfOpenPhysicalIndices += 2
            else
                indsRDM = vcat(-collect(1:numberOfOpenPhysicalIndices), 1, 2)
                indsB = [1, 3, -(numberOfOpenPhysicalIndices + 1)]
                indsK = [2, 3, -(numberOfOpenPhysicalIndices + 2)]
                reducedDensityMatrix = ncon([reducedDensityMatrix, finiteMPS[siteIdx],
                                             finiteMPS[siteIdx]], [indsRDM, indsB, indsK],
                                            [false, true, false])
            end

        else
            if any(openPositions .== siteIdx)
                indsRDM = vcat(-collect(1:numberOfOpenPhysicalIndices), 1, 2)
                indsB = [1, -(numberOfOpenPhysicalIndices + 2), 3]
                indsK = [2, -(numberOfOpenPhysicalIndices + 1), 3]
                reducedDensityMatrix = ncon([reducedDensityMatrix, finiteMPS[siteIdx],
                                             finiteMPS[siteIdx]], [indsRDM, indsB, indsK],
                                            [false, true, false])
                numberOfOpenPhysicalIndices += 2
            else
                indsRDM = vcat(-collect(1:numberOfOpenPhysicalIndices), 1, 2)
                indsB = [1, 3, 4]
                indsK = [2, 3, 4]
                reducedDensityMatrix = ncon([reducedDensityMatrix, finiteMPS[siteIdx],
                                             finiteMPS[siteIdx]], [indsRDM, indsB, indsK],
                                            [false, true, false])
            end
        end
    end

    # permute indices correctly
    reducedDensityMatrix = permute(reducedDensityMatrix,
                                   Tuple(idx for idx in 1:2:numberOfOpenPhysicalIndices),
                                   Tuple(idx for idx in 2:2:numberOfOpenPhysicalIndices))
    return reducedDensityMatrix
end

function compute_mutual_information(momentumModes::Vector{Int64}, finiteMPS::SparseMPS;
                                    svCutOff::Float64 = 1e-12)

    #----------------------------------------------------------------------------
    # compute entanglement entropy between -k and +k and between [-k, +k] and the rest
    #----------------------------------------------------------------------------

    # initialize matrix for the mutual information
    mutualInformationMatrix = zeros(Float64, length(momentumModes), length(momentumModes))

    # compute reduced density matrix and self information for each mode
    storeReducedDensityMatrices = Vector{TensorMap{ComplexF64}}(undef,
                                                                length(momentumModes))
    for (siteIdx, momentumVal) in enumerate(momentumModes)
        reducedDensityMatrix = compute_arbitrary_bipartition(finiteMPS, [siteIdx])
        _, eigVals = tsvd(reducedDensityMatrix; trunc = truncbelow(svCutOff))
        eigVals /= tr(eigVals)
        entanglementEntropy = -real(tr(eigVals * log(eigVals)))
        storeReducedDensityMatrices[siteIdx] = reducedDensityMatrix
        mutualInformationMatrix[siteIdx, siteIdx] = entanglementEntropy
    end

    # compute mutual information between all pairs of modes
    for (siteIdxA, momentumValA) in enumerate(momentumModes)
        for (siteIdxB, momentumValB) in enumerate(momentumModes)
            if siteIdxA < siteIdxB
                openIndices = [siteIdxA, siteIdxB]
                reducedDensityMatrixA = storeReducedDensityMatrices[siteIdxA]
                reducedDensityMatrixB = storeReducedDensityMatrices[siteIdxB]
                reducedDensityMatrixC = compute_arbitrary_bipartition(finiteMPS,
                                                                      openIndices)
                _, eigValsA = tsvd(reducedDensityMatrixA; trunc = truncbelow(svCutOff))
                _, eigValsB = tsvd(reducedDensityMatrixB; trunc = truncbelow(svCutOff))
                _, eigValsC = tsvd(reducedDensityMatrixC; trunc = truncbelow(svCutOff))
                eigValsA /= tr(eigValsA)
                eigValsB /= tr(eigValsB)
                eigValsC /= tr(eigValsC)
                entEntA = -real(tr(eigValsA * log(eigValsA)))
                entEntB = -real(tr(eigValsB * log(eigValsB)))
                entEntC = -real(tr(eigValsC * log(eigValsC)))
                mutualInformation = (entEntA + entEntB - entEntC)
                mutualInformationMatrix[siteIdxA, siteIdxB] = mutualInformation
                mutualInformationMatrix[siteIdxB, siteIdxA] = mutualInformation
            end
        end
    end

    # function return
    return mutualInformationMatrix
end
