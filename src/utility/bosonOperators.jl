include("vectorSpaces.jl")

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

function getDisplacementOperator(nMax::Int64, α::Number)
    """
    Construct displacement operator:
    D(z) = e^{-|z|^2 / 2} . e^{α a†} . e^{-α^* a}
    """
    displacementOp = exp(-abs(α)^2 / 2) * exp(α * getCreationOperator(nMax)) *
        exp(-conj(α) * getAnnihilationOperator(nMax))
    return displacementOp
end

function localAnnihilationOp(
        k::Int64, physVecSpace::ElementarySpace,
        conserveZ2::Bool = false
    )
    """ Construct a(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(-k => 1) : Rep[U₁ × ℤ₂]((-k, 1) => 1)
    annihilationOperator = zeros(
        ComplexF64, dimPhyVecSpace, dimPhyVecSpace,
        dim(auxVecSpace)
    )
    annihilationOperator[:, :, 1] = getAnnihilationOperator(dimPhyVecSpace - 1)
    annihilationOperator = TensorMap(
        annihilationOperator, physVecSpace,
        physVecSpace ⊗ auxVecSpace
    )
    return annihilationOperator
end

function localCreationOp(k::Int64, physVecSpace::ElementarySpace, conserveZ2::Bool = false)
    """ Construct a(k)^dag for a momentum-conserving MPO """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(+k => 1) : Rep[U₁ × ℤ₂]((+k, 1) => 1)
    creationOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    creationOperator[:, :, 1] = getCreationOperator(dimPhyVecSpace - 1)
    creationOperator = TensorMap(
        creationOperator, physVecSpace,
        physVecSpace ⊗ auxVecSpace
    )
    return creationOperator
end

function localNumberOp(physVecSpace::ElementarySpace, conserveZ2::Bool = false)
    """ Construct n(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    numberOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    numberOperator[:, :, 1] = getNumberOperator(dimPhyVecSpace - 1)
    numberOperator = TensorMap(
        numberOperator, physVecSpace,
        physVecSpace ⊗ auxVecSpace
    )
    return numberOperator
end

function localIdentityOp(physVecSpace::ElementarySpace, conserveZ2::Bool = false)
    """ Construct I for a momentum-conserving MPO """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    if spacetype(physVecSpace) == ComplexSpace
        auxVecSpace = !conserveZ2 ? ComplexSpace(1) : Rep[ℤ₂](0 => 1)
    elseif spacetype(physVecSpace) == U1Space
        auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    end
    identityOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    identityOperator[:, :, 1] = diagm(ones(dimPhyVecSpace))
    identityOperator = TensorMap(
        identityOperator, physVecSpace,
        physVecSpace ⊗ auxVecSpace
    )
    return identityOperator
end

function localMomentumOp(k::Int64, physVecSpace::ElementarySpace, conserveZ2::Bool = false)
    """ Construct k * n(k) for a momentum-conserving MPO """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    momentumOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    momentumOperator[:, :, 1] = k * getNumberOperator(dimPhyVecSpace - 1)
    momentumOperator = TensorMap(
        momentumOperator, physVecSpace,
        physVecSpace ⊗ auxVecSpace
    )
    return momentumOperator
end

function localParityOperator(
        k::Int64, physVecSpace::ElementarySpace,
        conserveZ2::Bool = false
    )
    """ Construct exp(iπ n(k)) for a momentum-conserving MPO """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    parityOp = exp(1im * π * getNumberOperator(dimPhyVecSpace - 1))
    parityOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    if k == 0 && conserveZ2
        rowColPerm = vcat(collect(1:2:dim(physVecSpace)), collect(2:2:dim(physVecSpace)))
        parityOperator[:, :, 1] = parityOp[rowColPerm, rowColPerm]
    else
        parityOperator[:, :, 1] = parityOp
    end
    parityOperator = TensorMap(parityOperator, physVecSpace, physVecSpace ⊗ auxVecSpace)
    return parityOperator
end

function localDisplacementOperator(
        k::Int64, physVecSpace::ElementarySpace, α::Number,
        conserveZ2::Bool = false
    )
    """ Construct D(α) for a momentum-conserving MPO """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    # auxVecSpace = !conserveZ2 ? U1Space(0 => 1) : Rep[U₁ × ℤ₂]((0, 0) => 1)
    auxVecSpace = removeDegeneracyQN(fuse(physVecSpace, conj(flip(physVecSpace))))

    # get ordering of QNs in physVecSpace and auxVecSpace
    phyQNSectors = physVecSpace.dims
    auxQNSectors = auxVecSpace.dims
    phyVecSpaceOrdering = [productSector.charge for productSector in keys(phyQNSectors)]
    auxVecSpaceOrdering = [productSector.charge for productSector in keys(auxQNSectors)]

    # create displacement operator
    displacementOp = getDisplacementOperator(dimPhyVecSpace - 1, α)
    displacementOperator = zeros(
        ComplexF64, dimPhyVecSpace, dimPhyVecSpace,
        dim(auxVecSpace)
    )
    for nBra in 0:(dim(physVecSpace) - 1), nKet in 0:(dim(physVecSpace) - 1)
        braIndPos = findfirst(phyVecSpaceOrdering .== (k * nBra))
        ketIndPos = findfirst(phyVecSpaceOrdering .== (k * nKet))
        auxIndPos = findfirst(auxVecSpaceOrdering .== (k * (nBra - nKet)))
        displacementOperator[braIndPos, ketIndPos, auxIndPos] = displacementOp[
            braIndPos,
            ketIndPos,
        ]
    end
    displacementOperator = TensorMap(
        displacementOperator, physVecSpace,
        physVecSpace ⊗ auxVecSpace
    )
    return displacementOperator
end

function localNegShiftOperator(physVecSpace::ElementarySpace, conserveZ2::Bool = false)
    """ Construct local operator for δ(n', n-1) """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? ComplexSpace(1) : Rep[ℤ₂](1 => 1)
    negShiftOp = LinearAlgebra.diagm(+1 => ones(dimPhyVecSpace - 1))
    negShiftOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    if conserveZ2
        rowColPerm = vcat(collect(1:2:dim(physVecSpace)), collect(2:2:dim(physVecSpace)))
        negShiftOperator[:, :, 1] = negShiftOp[rowColPerm, rowColPerm]
    else
        negShiftOperator[:, :, 1] = negShiftOp
    end
    negShiftOperator = TensorMap(negShiftOperator, physVecSpace, physVecSpace ⊗ auxVecSpace)
    return negShiftOperator
end

function localPosShiftOperator(physVecSpace::ElementarySpace, conserveZ2::Bool = false)
    """ Construct local operator for δ(n', n+1) """

    # get dimension of physVecSpace
    dimPhyVecSpace = dim(physVecSpace)
    auxVecSpace = !conserveZ2 ? ComplexSpace(1) : Rep[ℤ₂](1 => 1)
    posShiftOp = LinearAlgebra.diagm(-1 => ones(dimPhyVecSpace - 1))
    posShiftOperator = zeros(ComplexF64, dimPhyVecSpace, dimPhyVecSpace, dim(auxVecSpace))
    if conserveZ2
        rowColPerm = vcat(collect(1:2:dim(physVecSpace)), collect(2:2:dim(physVecSpace)))
        posShiftOperator[:, :, 1] = posShiftOp[rowColPerm, rowColPerm]
    else
        posShiftOperator[:, :, 1] = posShiftOp
    end
    posShiftOperator = TensorMap(posShiftOperator, physVecSpace, physVecSpace ⊗ auxVecSpace)
    return posShiftOperator
end

function mergeLocalOperators(localOpB::TensorMap, localOpT::TensorMap)
    """
    Apply localOpT onto localOpB to create a 3-leg MPO 
    with the spaces of the third legs fused 
    """
    fusionIsometry = isometry(
        space(localOpT, 3)' ⊗ space(localOpB, 3)',
        fuse(space(localOpT, 3)', space(localOpB, 3)')
    )
    @tensor newLocalOp[-1; -2 -3] := localOpT[1, -2, 2] * localOpB[-1, 1, 3] *
        fusionIsometry[2, 3, -3]
    return newLocalOp
end
