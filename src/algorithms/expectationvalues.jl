function computeExpValMPO(finiteMPS::SparseMPS, finiteMPO::SparseMPO)
    """ Computes expectation value for MPS and MPO """

    # # normalize finiteMPS
    # finiteMPS = normalizeMPS(finiteMPS);

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

function computeEnergyVariance(finiteMPS::SparseMPS, finiteMPO::SparseMPO)

    # copy finiteMPS
    myMPS = copy(finiteMPS);

    # compute H|ψ⟩
    maxBondDim = 3000;
    finiteMPOMPS = applyMPO(finiteMPO, myMPS, maxDim = maxBondDim, truncErr = 1e-6, compressionAlg = "zipUp");

    # compute E = ⟨ψ|H|ψ⟩
    mpsEnergy = dotMPS(myMPS, finiteMPOMPS);

    # compute |ϕ⟩ = H|ψ⟩ - E|ψ⟩
    myMPS[1] *= -mpsEnergy;
    phiMPS = finiteMPOMPS + myMPS;

    # compute variance as ⟨ϕ|ϕ⟩
    energyVariance = abs(real(dotMPS(phiMPS, phiMPS)));
    return energyVariance;

end