function expectation_value_mpo(finiteMPS::SparseMPS, finiteMPO::SparseMPO)
    """ Computes expectation value for MPS and MPO """

    # contract from left to right
    boundaryL = TensorMap(ones, space(finiteMPS[1], 1), space(finiteMPO[1], 1) ⊗ space(finiteMPS[1], 1));
    for siteIdx = 1 : +1 : length(finiteMPS)
        @tensor boundaryL[-1; -2 -3] := boundaryL[1, 3, 5] * finiteMPS[siteIdx][5, 4, -3] * finiteMPO[siteIdx][3, 2, -2, 4] * conj(finiteMPS[siteIdx][1, 2, -1]);
    end
    boundaryR = TensorMap(ones, space(finiteMPS[end], 3)' ⊗ space(finiteMPO[end], 3)', space(finiteMPS[end], 3)');

    # contact to get expectation value
    expectationVal = @tensor boundaryL[1, 2, 3] * boundaryR[3, 2, 1];
    return expectationVal;
    
end

function variance_mpo(finiteMPS::SparseMPS, finiteMPO::SparseMPO)

    # copy finiteMPS
    myMPS = copy(finiteMPS);

    # compute H|ψ⟩
    maxBondDim = 3000;
    finiteMPOMPS = applyMPO(finiteMPO, myMPS, maxDim = maxBondDim, truncErr = 1e-6, compressionAlg = "zipUp");

    # compute E = ⟨ψ|H|ψ⟩
    mpsEnergy = dotMPS(myMPS, finiteMPOMPS);

    # compute |ϕ⟩ = H|ψ⟩ - E|ψ⟩
    phiMPS = finiteMPOMPS - mpsEnergy * myMPS;

    # compute variance as ⟨ϕ|ϕ⟩
    energyVariance = abs(real(dotMPS(phiMPS, phiMPS)));
    return energyVariance;

end

function expectation_values(finiteMPS::SparseMPS, onsiteOperators::Vector{TensorMap})
    """ Computes the expectation value < psi | onsiteOp | psi > for all sites of the MPS """

    # get length of finiteMPS
    N = length(finiteMPS);

    # compute expectation values
    expVals = zeros(Float64, N);
    for siteIdx = 1 : N

        # bring MPS into canonical form
        orthogonalizeMPS!(finiteMPS, siteIdx);
        psiNormSq = real(tr(finiteMPS[siteIdx]' * finiteMPS[siteIdx]));
        
        # compute expectation value
        onsiteOp = onsiteOperators[siteIdx];
        expVal = @tensor conj(finiteMPS[siteIdx][1, 2, 4]) * onsiteOp[2, 3] * finiteMPS[siteIdx][1, 3, 4];
        expVals[siteIdx] = real(expVal) / psiNormSq;
    
    end
    return expVals;

end