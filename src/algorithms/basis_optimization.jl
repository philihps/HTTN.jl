function convertLocalOperatorsToTwoBodyGate(localOperators::Vector{<:AbstractTensorMap})
    kroneckerTensor = ones(ComplexF64, space(localOperators[1], 3)',
                           space(localOperators[2], 3))
    @tensor twoBodyGate[-1 -2; -3 -4] := localOperators[1][-1, -3, 1] *
                                         localOperators[2][-2, -4, 2] *
                                         kroneckerTensor[1, 2]
    return twoBodyGate
end

function applyTwoModeTransformation(twoModeU::AbstractTensorMap,
                                    twoModeT::AbstractTensorMap)
    @tensor twoModeUT[-1 -2 -3; -4] := twoModeU[-2, -3, 2, 3] * twoModeT[-1, 2, 3, -4]
    return twoModeUT
end

function computeRenyiEntropy(twoSiteTensor::TensorMap)
    _, S, _ = tsvd(twoSiteTensor, ((1, 2), (3, 4)))
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
function squeezingOp(ξ::Union{Int64,Float64,ComplexF64},
                     nMax::Int64,
                     kL::Int64,
                     kR::Int64,
                     PL::ElementarySpace,
                     PR::ElementarySpace)
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
    NuId = convertLocalOperatorsToTwoBodyGate([locaNumberOp(PL), localIdentityOp(PR)])
    IdNu = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), locaNumberOp(PR)])

    K_A_0 = 1 / 2 * (NuId + IdNu + IdId)
    K_A_min = AnAn
    K_A_plus = CrCr

    # construct squeezing operator
    S = matrixExponentialSeries(-1 * tanh(ξ) * K_A_plus, nMax) *
        matrixExponentialSeries(-2 * log(cosh(ξ)) * K_A_0, nMax) *
        matrixExponentialSeries(tanh(ξ) * K_A_min, nMax)
    return S
end

function singleSqueezingOp(ξ::Union{Int64,Float64,ComplexF64},
                           nMax::Int64,
                           physSpace::ElementarySpace)
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
    K_A_min = (1 / 2) * An
    K_A_plus = (1 / 2) * Cr

    # construct squeezing operator
    S = matrixExponentialSeries(-1 * tanh(ξ) * K_A_plus, nMax) *
        matrixExponentialSeries(-2 * log(cosh(ξ)) * K_A_0, nMax) *
        matrixExponentialSeries(1 * tanh(ξ) * K_A_min, nMax)
    return S
end

# function singleSqueezingOp(ξ::Union{Int64,Float64,ComplexF64},
#     nMax::Int64,
#     physSpace::ElementarySpace)
#     """
#     Squeezing operator for zero mode

#     S = e^{-tanh(ξ) . K_A_plus} . e^{-2 . log(cosh(ξ)) . K_A_0} . e^{tanh(ξ) . K_A_min}
#     K_A_0 = (1/2) * (N_{-k} + N_k + Id)
#     K_A_plus = Cr_k . Cr_{-k}
#     K_A_min = An_{-k} . An_k
#     """

#     # construct one-site operators
#     Cr = TensorMap(getCreationOperator(nMax), physSpace, physSpace)
#     @tensor CrCr[-1; -2] := Cr[-1, 1] * Cr[1, -2]
#     An = TensorMap(getAnnihilationOperator(nMax), physSpace, physSpace)
#     @tensor AnAn[-1; -2] := An[-1, 1] * An[1, -2]
#     numberOp = TensorMap(getNumberOperator(nMax), physSpace, physSpace)
#     idOp = TensorMap(getIdentityOperator(dim(physSpace)), physSpace, physSpace)

#     K_A_0 = numberOp + (1 / 2) * idOp
#     K_A_min = AnAn
#     K_A_plus = CrCr

#     # construct squeezing operator
#     S = matrixExponentialSeries(-1 * tanh(ξ) * K_A_plus, nMax) *
#     matrixExponentialSeries(-2 * log(cosh(ξ)) * K_A_0, nMax) *
#     matrixExponentialSeries(tanh(ξ) * K_A_min, nMax)
#     return S
# end

function findDisentanglingRotation(ξ::Union{Int64,Float64,ComplexF64},
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

# function value_and_gradient(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace, twoSiteTensor::TensorMap)
#     fval = findDisentanglingRotation(ξ, nMax, kL, kR, PL, PR, twoSiteTensor);
#     gval = Zygote.gradient(x -> findDisentanglingRotation(x, nMax, kL, kR, PL, PR, twoSiteTensor), ξ)[1];
#     return real(fval), gval;
# end

# # OptimKit functions for optimization of Float64 values
# _scale!(v, α) = v * α;
# _add!(vdst, vsrc, α) = vdst += vsrc * α;

#---------------------------------------------------------------------------
# analytic gradient for basis optimization

function analyticGradientCostFunction(ξ::Union{Int64,Float64,ComplexF64},
                                      nMax::Int64,
                                      kL::Int64,
                                      kR::Int64,
                                      PL::ElementarySpace,
                                      PR::ElementarySpace,
                                      twoSiteTensor::TensorMap)
    sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR)
    SPsi = applyTwoModeTransformation(sqOp, twoSiteTensor)
    U, S, V = tsvd(SPsi, ((1, 2), (3, 4)))
    dPsidXi = gradient_squeezed_tensor(ξ, nMax, kL, kR, PL, PR, twoSiteTensor)
    dPsidXi = permute(dPsidXi, ((1, 2), (3, 4)))
    dSVdXi = real(gradient_singular_value_matrix(U, dPsidXi, V))
    analyticGradient = 2 / tr(S) * tr(dSVdXi)
    return analyticGradient
end

function gradient_singular_value_matrix(U::TensorMap, dAdXi::TensorMap, V::TensorMap)
    return 0.5 * (U' * dAdXi * V' + V * dAdXi' * U)
end

# dμ(ξ) = sign(ξ) * sinh(abs(ξ));
# dν(ξ) = cosh(abs(ξ));
dμ(ξ) = sinh(ξ);
dν(ξ) = cosh(ξ);

function gradient_squeezing_operator(ξ::Union{Int64,Float64,ComplexF64},
                                     nMax::Int64,
                                     kL::Int64,
                                     kR::Int64,
                                     PL::ElementarySpace,
                                     PR::ElementarySpace)

    # construct two-site operators
    CrCr = convertLocalOperatorsToTwoBodyGate([localCreationOp(kL, PL),
                                               localCreationOp(kR, PR)])
    AnAn = convertLocalOperatorsToTwoBodyGate([localAnnihilationOp(kL, PL),
                                               localAnnihilationOp(kR, PR)])
    IdId = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localIdentityOp(PR)])
    NuId = convertLocalOperatorsToTwoBodyGate([locaNumberOp(PL), localIdentityOp(PR)])
    IdNu = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), locaNumberOp(PR)])

    # compute μ and ν
    μ = cosh(ξ)
    ν = sinh(ξ)

    # construct K0, K1 and K2
    # K0 = -log(μ) * (NuId + IdNu + IdId);
    # K1 = conj(ν)/μ * AnAn;
    # K2 = -ν/μ * CrCr;
    K0 = -log(μ) * (NuId + IdNu + IdId)
    K1 = tanh(ξ) * AnAn
    K2 = -tanh(ξ) * CrCr

    # construct squeezing operator
    A = exp(K2)
    B = exp(K0)
    C = exp(K1)
    S = exp(K2) * exp(K0) * exp(K1)

    # construct derivative of squeezing operator
    dA = -1 * CrCr * (dν(ξ) / μ - ν * dμ(ξ) / μ^2)
    dB = -1 * (NuId + IdNu + IdId) * dμ(ξ) / μ
    dC = +1 * AnAn * (dν(ξ) / μ - ν * dμ(ξ) / μ^2)
    dSdXi = A * dA * B * C + A * B * dB * C + A * B * C * dC
    # # dSdXi = S * AnAn - CrCr * S;
    # dSdXi = S * (AnAn - CrCr)
    return dSdXi
end

function gradient_squeezed_tensor(ξ::Union{Int64,Float64,ComplexF64},
                                  nMax::Int64,
                                  kL::Int64,
                                  kR::Int64,
                                  PL::ElementarySpace,
                                  PR::ElementarySpace,
                                  twoSiteTensor::TensorMap)
    dSdXi = gradient_squeezing_operator(ξ, nMax, kL, kR, PL, PR)
    dPsiXi = applyTwoModeTransformation(dSdXi, twoSiteTensor)
    return dPsiXi
end

#---------------------------------------------------------------------------
