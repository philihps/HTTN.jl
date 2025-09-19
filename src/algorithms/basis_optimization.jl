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
    CrCr = convertLocalOperatorsToTwoBodyGate([localCreationOp(kL, PL),
                                               localCreationOp(kR, PR)])
    AnAn = convertLocalOperatorsToTwoBodyGate([localAnnihilationOp(kL, PL),
                                               localAnnihilationOp(kR, PR)])
    IdId = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localIdentityOp(PR)])
    NuId = convertLocalOperatorsToTwoBodyGate([localNumberOp(PL), localIdentityOp(PR)])
    IdNu = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localNumberOp(PR)])

    # construct K operators
    K_A_0 = 1 / 2 * (NuId + IdNu + IdId)
    K_A_min = AnAn
    K_A_plus = CrCr

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