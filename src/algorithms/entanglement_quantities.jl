
function LinearAlgebra.diag(T::AbstractTensorMap)
    """ Overloading of LinearAlgebra function diag for TensorMap type """

    diagElements = Vector{eltype(T)}();
    blockSectors = blocksectors(T);
    for blockIdx = blockSectors
        append!(diagElements, diag(block(T, blockIdx)))
    end
    return diagElements;

end

function compute_entanglement_spectra(finiteMPS::SparseMPS; svCutOff::Float64 = 1e-12)
    """ Returns a vector of entanglement spectra for successive bipartitions of the MPS """

    # initialize list of entanglement spectra
    entanglementSpectra = Vector{Vector{Float64}}(undef, length(finiteMPS) - 1);

    # compute entanglement spectra by two-site SVDs
    for siteIdx = 1 : (length(finiteMPS) - 1)

        # orthogonalize MPS
        orthogonalizeMPS!(finiteMPS, siteIdx);

        # construct initial theta
        theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(theta, (1, 2), (3, 4), trunc = truncbelow(svCutOff));
        S /= norm(S);
        entanglementSpectra[siteIdx] = real.(diag(S));

    end
    return entanglementSpectra;

end

function compute_entanglement_entropies(finiteMPS::SparseMPS; svCutOff::Float64 = 1e-12)
    """ Compute von-Neumann entropy for a each bipartition along the MPS """

    entanglementSpectra = compute_entanglement_spectra(finiteMPS, svCutOff = svCutOff);
    entEntropy = abs.([-sum(entSpec.^2 .* log.(entSpec.^2)) for entSpec in entanglementSpectra]);
    return entEntropy;

end