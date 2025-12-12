function convertSqueezingParameter(ξ::Number)
    """
    Convert squeezing parameter ξ to μ and ν.
    """
    μ = cosh(abs(ξ))
    ν = exp(1im * angle(ξ)) * sinh(abs(ξ))
    return μ, ν
end

function convertLocalOperatorsToTwoBodyGate(localOperators::Vector{<:AbstractTensorMap})
    kroneckerTensor = ones(ComplexF64, space(localOperators[1], 3)',
                           space(localOperators[2], 3))
    @tensor twoBodyGate[-1 -2; -3 -4] := localOperators[1][-1, -3, 1] *
                                         localOperators[2][-2, -4, 2] *
                                         kroneckerTensor[1, 2]
    return twoBodyGate
end

function applyTwoModeTransformation(twoModeU::TensorMap,
                                    twoModeT::TensorMap)
    @tensor twoModeUT[-1 -2 -3; -4] := twoModeU[-2, -3, 2, 3] * twoModeT[-1, 2, 3, -4]
    return twoModeUT
end

function computeRenyiEntropy(twoSiteTensor::TensorMap)
    _, S, _ = tsvd(twoSiteTensor, ((1, 2), (3, 4)); alg = TensorKit.SVD())
    return 2 * log(tr(S))
end

function computeEntropy(twoSiteTensor::TensorMap)
    _, S, _ = tsvd(twoSiteTensor, ((1, 2), (3, 4)))
    vnEntropy = abs(-tr(S^2 * log(S^2)))
    return vnEntropy
end

function matrixExponentialSeries(operator::TensorMap, nMax::Int64)
    matrixExp = sum([1 / Float64(factorial(big(n))) * operator^n for n in collect(0:nMax)])
    return matrixExp
end

# construct squeezing operator
function squeezingOp(ξ::Number,
                     nMax::Int64,
                     kL::Int64,
                     kR::Int64,
                     PL::ElementarySpace,
                     PR::ElementarySpace;
                     conserveZ2::Bool = false)
    """
    S = e^{-tanh(ξ) . K_A_plus} . e^{-2 . log(cosh(ξ)) . K_A_0} . e^{tanh(ξ) . K_A_min}
    K_A_0 = (1/2) * (N_{-k} + N_k + Id)
    K_A_plus = Cr_k . Cr_{-k}
    K_A_min = An_{-k} . An_k
    """

    # construct two-site operators

    CrCr = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localCreationOp(kL, PL),
                                               localCreationOp(kR, PR)])
    AnAn = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localAnnihilationOp(kL, PL),
                                               localAnnihilationOp(kR, PR)])
    IdId = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localIdentityOp(PR)])
    NuId = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localNumberOp(PL), localIdentityOp(PR)])
    IdNu = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localNumberOp(PR)])

    # # construct K operators
    # K_A_0 = 1 / 2 * (NuId + IdNu + IdId)
    # K_A_min = AnAn
    # K_A_plus = CrCr

    # compute μ and ν
    μ, ν = convertSqueezingParameter(ξ)

    S = exp(conj(ξ) * AnAn - ξ * CrCr)
    # S = matrixExponentialSeries(conj(ξ) * AnAn - ξ * CrCr, nMax)

    # construct squeezing operator
    # S = exp(-1 * tanh(ξ) * K_A_plus) * exp(-2 * log(cosh(ξ)) * K_A_0) * exp(tanh(ξ) * K_A_min)
    # S = exp(-1 * ν / μ * K_A_plus) * exp(-2 * log(μ) * K_A_0) * exp(conj(ν) / μ * K_A_min)
    # S = matrixExponentialSeries(-1 * ν / μ * K_A_plus, nMax) *
    #     matrixExponentialSeries(-2 * log(μ) * K_A_0, nMax) *
    #     matrixExponentialSeries(conj(ν) / μ * K_A_min, nMax)
    return S
end

function singleSqueezingOp(ξ::Number,
                           nMax::Int64,
                           physSpace::ElementarySpace;
                           conserveZ2::Bool = false)
    """
    Squeezing operator for zero mode
        
    S = e^{-tanh(ξ) . K_A_plus} . e^{-2 . log(cosh(ξ)) . K_A_0} . e^{tanh(ξ) . K_A_min}
    K_A_0 = (1/2) * (N_{-k} + N_k + Id)
    K_A_plus = Cr_k . Cr_{-k}
    K_A_min = An_{-k} . An_k
    """

    # construct one-site operators
    Cr = TensorMap(getCreationOperator(nMax), physSpace, physSpace)
    An = TensorMap(getAnnihilationOperator(nMax), physSpace, physSpace)
    numberOp = TensorMap(getNumberOperator(nMax), physSpace, physSpace)
    idOp = TensorMap(getIdentityOperator(dim(physSpace)), physSpace, physSpace)

    K_A_0 = (1 / 2) * numberOp + 1 / 4 * idOp
    K_A_min = (1 / 2) * An * An
    K_A_plus = (1 / 2) * Cr * Cr

    # construct squeezing operator
    S = matrixExponentialSeries(-1 * tanh(ξ) * K_A_plus, nMax) *
        matrixExponentialSeries(-2 * log(cosh(ξ)) * K_A_0, nMax) *
        matrixExponentialSeries(1 * tanh(ξ) * K_A_min, nMax)
    return S
end

function findDisentanglingRotation(ξ::Number,
                                   nMax::Int64,
                                   kL::Int64,
                                   kR::Int64,
                                   PL::ElementarySpace,
                                   PR::ElementarySpace,
                                   twoSiteTensor::TensorMap)
    twoSiteSqueezingOperator = squeezingOp(ξ, nMax, kL, kR, PL, PR)
    costFunction = computeRenyiEntropy(applyTwoModeTransformation(twoSiteSqueezingOperator,
                                                                  twoSiteTensor))
    return costFunction
end

function value_and_gradient(ξ::Number, nMax::Int64, kL::Int64, kR::Int64,
                            PL::ElementarySpace, PR::ElementarySpace,
                            twoSiteTensor::TensorMap)
    fval = findDisentanglingRotation(ξ, nMax, kL, kR, PL, PR, twoSiteTensor)
    gval = Zygote.gradient(x -> findDisentanglingRotation(x, nMax, kL, kR, PL, PR,
                                                          twoSiteTensor), ξ)[1]
    return fval, gval
end

function bogTransformMPO(finiteMPO::SparseMPO, bogParameters::Union{Vector{Float64}, Vector{ComplexF64}}; truncErr::Float64 = 1e-8)

    # bring MPO into canonical form
    finiteMPO = orthogonalizeMPO(finiteMPO, 1)

    # compute initial norm of the MPO
    initialNorm = real(tr(finiteMPO[1]' * finiteMPO[1]))

    # loop over all sites and apply squeezing transformation
    truncationErrors = Float64[]
    for siteIdx in 1:+1:(length(finiteMPO) - 1)

        # apply squeezing transformation
        if mod(siteIdx, 2) == 0

            # construct two-site MPO tensor
            @tensor AC2[-1 -2 -3; -4 -5 -6] := finiteMPO[siteIdx][-1, -2, 1, -4] * finiteMPO[siteIdx + 1][1, -3, -6, -5]

            # get physVecSpaces for squeezing operator
            PL = space(finiteMPO[siteIdx + 0], 2)
            PR = space(finiteMPO[siteIdx + 1], 2)
            nMax = Int(0.5 * (dim(PL) - 1 + dim(PR) - 1))

            # get original bond dimension between the two sites
            dimVirtSpace = dim(space(finiteMPO[siteIdx + 1], 1))

            # set kL and kR
            kL = -1 * Int(siteIdx / 2)
            kR = +1 * Int(siteIdx / 2)
            # display([kL  kR])

            # construct two-site squeezing operator and transform two-site MPO tensor
            squeezingOperator = squeezingOp(bogParameters[1 + kR], nMax, kL, kR, PL, PR)
            @tensor newAC2[-1 -2 -3; -4 -5 -6] := squeezingOperator[-2, -3, 2, 3] * AC2[-1, 2, 3, 4, 5, -6] * squeezingOperator'[4, 5, -4, -5]

            # perform SVD and truncate to desired bond dimension (also move orthogonality center to the right)
            U, S, V, ϵ = tsvd(newAC2, ((1, 2, 4), (3, 6, 5)); trunc = truncdim(dimVirtSpace) & truncerr(truncErr), alg = TensorKit.SVD(),)
            # S /= norm(S)
            finiteMPO[siteIdx + 0] = permute(U, ((1, 2), (4, 3)))
            finiteMPO[siteIdx + 1] = permute(S * V, ((1, 2), (3, 4)))
            truncationErrors = vcat(truncationErrors, ϵ)

        else

            (Q, R) = leftorth(finiteMPO[siteIdx], ((4, 1, 2), (3,)); alg = QRpos())
            finiteMPO[siteIdx + 0] = permute(Q, ((2, 3), (4, 1)))
            finiteMPO[siteIdx + 1] = permute(R * permute(finiteMPO[siteIdx + 1], ((1,), (2, 3, 4))), ((1, 2), (3, 4)))

        end

    end

    # loop over all sites and restore right-canonical form
    for siteIdx in length(finiteMPO):-1:2
        (L, Q) = rightorth(finiteMPO[siteIdx], ((1,), (2, 3, 4)); alg = LQpos())
        finiteMPO[siteIdx - 1] = permute(permute(finiteMPO[siteIdx - 1], ((4, 1, 2), (3,))) * L, ((2, 3), (4, 1)))
        finiteMPO[siteIdx - 0] = permute(Q, ((1, 2), (3, 4)))
    end

    # compute final norm and renormalize the MPO
    finalNorm = real(tr(finiteMPO[1]' * finiteMPO[1]))
    finiteMPO[1] *= sqrt(initialNorm / finalNorm)

    return finiteMPO, truncationErrors

end