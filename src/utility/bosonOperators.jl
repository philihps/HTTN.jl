
function getIdentityOperator(dimHilbertSpace::Int64)
    bosonOp = LinearAlgebra.diagm(ones(Float64, dimHilbertSpace))
    return bosonOp
end

function getNumberOperator(numBosons::Int64)
    bosonOp = LinearAlgebra.diagm(collect(0:numBosons))
    return bosonOp
end

function getAnnihilationOperator(numBosons::Int64)
    bosonOp = -1im * LinearAlgebra.diagm(+1 => sqrt.(collect(1:numBosons)))
    return bosonOp
end

function getCreationOperator(numBosons::Int64)
    bosonOp = +1im * LinearAlgebra.diagm(-1 => sqrt.(collect(1:numBosons)))
    return bosonOp
end

function expMatrices(oP, numBosons::Int64, z::Union{Int64,Float64,ComplexF64})
    """
    Compute e^{z . oP}
    """
    Id = getIdentityOperator(numBosons + 1)
    opPowers = zeros(ComplexF64, numBosons + 1, numBosons + 1, numBosons + 1)
    opPowers[1, :, :] = Id
    for n in 1:numBosons
        opPowers[n + 1, :, :] = z * oP * opPowers[n, :, :]
    end
    expOp = sum(1 / gamma(n + 1) * opPowers[n + 1, :, :] for n in collect(0:(numBosons - 1)))

    return expOp

end

function getDisplacementOperator(nMax::Int64, α::Number)
    """
    Construct displacement operator:
    D(z) = e^{-|z|^2 / 2} . e^{α a†} . e^{-α* a}
    """
    # displacementOp = exp(-abs(α)^2 / 2) * exp(α * getCreationOperator(nMax)) *
    #                  exp(-conj(α) * getAnnihilationOperator(nMax))
    displacementOp = exp(-abs(α)^2 / 2) * expMatrices(getCreationOperator(nMax), nMax, α) *
                     expMatrices(getAnnihilationOperator(nMax), nMax, -conj(α))
    return displacementOp
end

function localAnnihilationOp(k::Int64, physVecSpace::GradedSpace)
    """ Construct a(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = U1Space(-k => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = getAnnihilationOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function localCreationOp(k::Int64, physVecSpace::GradedSpace)
    """ Construct a(k)^dag for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = U1Space(+k => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = getCreationOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function locaNumberOp(physVecSpace::GradedSpace)
    """ Construct n(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = U1Space(0 => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = getNumberOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function localIdentityOp(physVecSpace::GradedSpace)
    """ Construct I for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = U1Space(0 => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = diagm(ones(dimPhyVecSpace))
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function localMomentumOp(k::Int64, physVecSpace::GradedSpace)
    """ Construct k * n(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = U1Space(0 => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = k * getNumberOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end
