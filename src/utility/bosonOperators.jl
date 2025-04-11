
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

function mergeLocalOperators(localOpB::TensorMap, localOpT::TensorMap)
    """
    Apply localOpT onto localOpB to create a 3-leg MPO 
    with the spaces of the third legs fused 
    """
    fusionIsometry = isometry(space(localOpT, 3)' ⊗ space(localOpB, 3)',
                              fuse(space(localOpT, 3)', space(localOpB, 3)'))
    @tensor newLocalOp[-1; -2 -3] := localOpT[1, -2, 2] * localOpB[-1, 1, 3] *
                                     fusionIsometry[2, 3, -3]
    return newLocalOp
end

function getDisplacementOperator(nMax::Int64, α::Number)
    """
    Construct displacement operator:
    D(z) = e^{-|z|^2 / 2} . e^{α a†} . e^{-α* a}
    """
    displacementOp = exp(-abs(α)^2 / 2) * exp(α * getCreationOperator(nMax)) *
                     exp(-conj(α) * getAnnihilationOperator(nMax))
    return displacementOp
end

function localAnnihilationOp(k::Int64, physVecSpace::GradedSpace, conserveZ2::Bool = false)
    """ Construct a(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(-k => 1) : Rep[U₁ × ℤ₂]((-k, 0) => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = getAnnihilationOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function localCreationOp(k::Int64, physVecSpace::GradedSpace, conserveZ2::Bool = false)
    """ Construct a(k)^dag for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(+k => 1) : Rep[U₁ × ℤ₂]((+k, 0) => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = getCreationOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function locaNumberOp(physVecSpace::GradedSpace, conserveZ2::Bool = false)
    """ Construct n(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = getNumberOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function localIdentityOp(physVecSpace::GradedSpace, conserveZ2::Bool = false)
    """ Construct I for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = diagm(ones(dimPhyVecSpace))
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function localMomentumOp(k::Int64, physVecSpace::GradedSpace, conserveZ2::Bool = false)
    """ Construct k * n(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    interactionTensor = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    interactionTensor[:, :, 1] = k * getNumberOperator(dimPhyVecSpace - 1)
    interactionTensor = TensorMap(interactionTensor, physVecSpace,
                                  physVecSpace ⊗ auxVecSpace)
    return interactionTensor
end

function localParityOperator(k::Int64, physVecSpace::GradedSpace, conserveZ2::Bool = false)
    """ Construct exp(iπ n(k)) for a momentum-conserving MPO """

    # get dimension of physVecSpace 
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    numberOp = exp(1im * π * getNumberOperator(dimPhyVecSpace - 1))
    numberOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    if k == 0 && conserveZ2
        rowColPerm = vcat(collect(1 : 2 : dim(physVecSpace)), collect(2 : 2 : dim(physVecSpace)))
        numberOperator[:, :, 1] = numberOp[rowColPerm, rowColPerm]
    else
        numberOperator[:, :, 1] = numberOp
    end
    numberOperator = TensorMap(numberOperator, physVecSpace, physVecSpace ⊗ auxVecSpace)
    return numberOperator
end