function expectation_value_mpo(finiteMPS, finiteMPO)
    """ Computes expectation value for MPS and MPO """

    # contract from left to right
    boundaryL = ones(ComplexF64, space(finiteMPS[1], 1),
                     space(finiteMPO[1], 1) ⊗ space(finiteMPS[1], 1))
    for siteIdx in 1:+1:length(finiteMPS)
        @tensor boundaryL[-1; -2 -3] := boundaryL[1, 3, 5] *
                                        finiteMPS[siteIdx][5, 4, -3] *
                                        finiteMPO[siteIdx][3, 2, -2, 4] *
                                        conj(finiteMPS[siteIdx][1, 2, -1])
    end
    boundaryR = ones(ComplexF64,
                     space(finiteMPS[end], 3)' ⊗ space(finiteMPO[end], 3)',
                     space(finiteMPS[end], 3)')

    # contact to get expectation value
    expectationVal = @tensor boundaryL[1, 2, 3] * boundaryR[3, 2, 1]
    return expectationVal
end

function variance_mpo(finiteMPS::SparseMPS, finiteMPO::SparseMPO)

    # copy finiteMPS
    myMPS = copy(finiteMPS)

    # compute H|ψ⟩
    finiteMPOMPS = applyMPO(finiteMPO, myMPS; truncErr = 1e-4, compressionAlg = "zipUp")

    # compute E = ⟨ψ|H|ψ⟩
    mpsEnergy = dotMPS(myMPS, finiteMPOMPS)

    # compute |ϕ⟩ = H|ψ⟩ - E|ψ⟩
    phiMPS = finiteMPOMPS - mpsEnergy * myMPS

    # compute variance as ⟨ϕ|ϕ⟩
    energyVariance = abs(real(dotMPS(phiMPS, phiMPS)))
    return energyVariance
end

function expectation_values(finiteMPS::SparseMPS,
                            onsiteOperators::Vector{TensorMap{ComplexF64}})
    """ Computes the expectation value < psi | onsiteOp | psi > for all sites of the MPS """

    fMPS = copy(finiteMPS)

    # compute expectation values
    expVals = zeros(Float64, length(fMPS))
    for siteIdx in 1:length(fMPS)

        # bring MPS into canonical form
        orthogonalizeMPS!(fMPS, siteIdx)
        psiNormSq = real(tr(fMPS[siteIdx]' * fMPS[siteIdx]))

        # compute expectation value
        onsiteOp = onsiteOperators[siteIdx]
        expVal = @tensor conj(fMPS[siteIdx][1, 2, 4]) *
                         onsiteOp[2, 3] *
                         fMPS[siteIdx][1, 3, 4]
        expVals[siteIdx] = real(expVal) / psiNormSq
    end
    return expVals
end

function expectation_values_density_matrix(thermalDensityMatrix::SparseMPO,
                                           finiteMPO::SparseMPO)
    """ Calculates expectation values for thermal states using an MPO """

    # get length of thermalDensityMatrix
    N = length(thermalDensityMatrix)

    # contract from left to right
    boundaryL = ones(space(thermalDensityMatrix[1], 1),
                     space(finiteMPO[1], 1) ⊗ space(thermalDensityMatrix[1], 1))
    for siteIdx in 1:+1:N
        @tensor boundaryL[-1; -2 -3] := boundaryL[1, 3, 5] *
                                        thermalDensityMatrix[siteIdx][5, 4, -3, 6] *
                                        finiteMPO[siteIdx][3, 2, -2, 4] *
                                        conj(thermalDensityMatrix[siteIdx][1, 2, -1, 6])
    end
    boundaryR = ones(space(thermalDensityMatrix[N], 3)' ⊗ space(finiteMPO[N], 3)',
                     space(thermalDensityMatrix[N], 3)')

    # contact to get expectation value
    expectationVal = @tensor boundaryL[1, 2, 3] * boundaryR[3, 2, 1]
    return real(expectationVal)
end
