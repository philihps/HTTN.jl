function convertLocalOperatorsToTwoBodyGate(localOperators::Vector{<:AbstractTensorMap})
    kroneckerTensor = TensorMap(ones, space(localOperators[1], 3)', space(localOperators[2], 3));
    @tensor twoBodyGate[-1 -2; -3 -4] := localOperators[1][-1, -3, 1] * localOperators[2][-2, -4, 2] * kroneckerTensor[1, 2];
    return twoBodyGate;
end

function applyTwoModeTransformation(twoModeU::AbstractTensorMap, twoModeT::AbstractTensorMap)
    @tensor twoModeUT[-1 -2 -3; -4] := twoModeU[-2, -3, 2, 3] * twoModeT[-1, 2, 3, -4];
    return twoModeUT;
end

function computeRenyiEntropy(twoSiteTensor::TensorMap)
    _, S, _ = tsvd(twoSiteTensor, (1, 2), (3, 4));
    return 2 * log(tr(S));
end

function computeEntropy(twoSiteTensor::TensorMap)
    _, S, _ = tsvd(twoSiteTensor, (1, 2), (3, 4));
    vnEntropy = abs(-tr(S^2 * log(S^2)));
    return vnEntropy;
end

function matrixExponentialSeries(operator::TensorMap, nMax::Int64)
    matrixExp = sum([1 / factorial(n) * operator^n for n in collect(0 : nMax)]);
    return matrixExp;
end

# construct squeezing operator
function squeezingOp(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace)

    # construct two-site operators
    CrCr = convertLocalOperatorsToTwoBodyGate([localCreationOp(kL, PL), localCreationOp(kR, PR)]);
    AnAn = convertLocalOperatorsToTwoBodyGate([localAnnihilationOp(kL, PL), localAnnihilationOp(kR, PR)]);
    IdId = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localIdentityOp(PR)]);
    NuId = convertLocalOperatorsToTwoBodyGate([locaNumberOp(PL), localIdentityOp(PR)]);
    IdNu = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), locaNumberOp(PR)]);

    # compute μ and ν
    # μ = cosh(abs(ξ));
    # ν = exp(1im * angle(ξ)) * sinh(abs(ξ));
    # compute μ and ν
    # μ = cosh(abs(ξ));
    # ν = sign(ξ) * sinh(abs(ξ));
    # μ = cosh(ξ);
    # ν = sinh(ξ);

    # # construct K0, K1 and K2
    # K0 = -log(μ) * (NuId + IdNu + IdId);
    # K1 = conj(ν)/μ * AnAn;
    # K2 = -ν/μ * CrCr;
    K0 = -1 * log(cosh(ξ)) * (NuId + IdNu + IdId); 
    K1 = +1 * tanh(ξ) * AnAn; 
    K2 = -1 * tanh(ξ) * CrCr;

    # K0 = -log(μ) * (NuId + IdNu + IdId);
    # K1 = tanh(ξ) * AnAn;
    # K2 = -tanh(ξ) * CrCr;

    # # compute eigenvalue decomposition
    # D1, U1 = eigen(CrCr + AnAn);
    # D2, U2  = eig(NuId + IdNu + IdId);
    # D3, U2 = eig(AnAn);
    # display(CrCr)
    # matCrCr = convert(Array, CrCr);
    # matCrCr = reshape(matCrCr, size(matCrCr, 1) * size(matCrCr, 2), size(matCrCr, 3) * size(matCrCr, 4));
    # display(matCrCr)
    # display(real(D1))
    # display(norm(U1 * D1 * inv(U1) - (CrCr + AnAn)))

    # COMM = CrCr * AnAn - AnAn * CrCr;
    # COMM = K2 * K0 - K0 * K2;
    # matCOMM = convert(Array, COMM);
    # matCOMM = reshape(matCOMM, size(matCOMM, 1) * size(matCOMM, 2), size(matCOMM, 3) * size(matCOMM, 4));
    # display(matCOMM)

    # construct squeezing operator
    S = matrixExponentialSeries(K2, nMax) * matrixExponentialSeries(K0, nMax) * matrixExponentialSeries(K1, nMax);
    # S = matrixExponentialSeries(K2, min(nMax, 10)) * matrixExponentialSeries(K0, 10) * matrixExponentialSeries(K1, min(nMax, 10));
    # S = exp(K2) * exp(K0) * exp(K1);
    return S;

end

function findDisentanglingRotation(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace, twoSiteTensor::TensorMap)
    twoSiteSqueezingOperator = squeezingOp(ξ, nMax, kL, kR, PL, PR);
    costFunction = computeRenyiEntropy(applyTwoModeTransformation(twoSiteSqueezingOperator, twoSiteTensor));
    return costFunction;
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

function analyticGradientCostFunction(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace, twoSiteTensor::TensorMap)
    sqOp = squeezingOp(ξ, nMax, kL, kR, PL, PR);
    SPsi = applyTwoModeTransformation(sqOp, twoSiteTensor);
    U, S, V = tsvd(SPsi, (1, 2), (3, 4));
    dPsidXi = gradient_squeezed_tensor(ξ, nMax, kL, kR, PL, PR, twoSiteTensor);
    dPsidXi = permute(dPsidXi, (1, 2), (3, 4));
    dSVdXi = real(gradient_singular_value_matrix(U, dPsidXi, V));
    analyticGradient = 2 / tr(S) * tr(dSVdXi);
    return analyticGradient;
end

function gradient_singular_value_matrix(U::TensorMap, dAdXi::TensorMap, V::TensorMap)
    return 0.5 * (U' * dAdXi * V' + V * dAdXi' * U);
end

# dμ(ξ) = sign(ξ) * sinh(abs(ξ));
# dν(ξ) = cosh(abs(ξ));
dμ(ξ) = sinh(ξ);
dν(ξ) = cosh(ξ);

function gradient_squeezing_operator(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace)

    # construct two-site operators
    CrCr = convertLocalOperatorsToTwoBodyGate([localCreationOp(kL, PL), localCreationOp(kR, PR)]);
    AnAn = convertLocalOperatorsToTwoBodyGate([localAnnihilationOp(kL, PL), localAnnihilationOp(kR, PR)]);
    IdId = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localIdentityOp(PR)]);
    NuId = convertLocalOperatorsToTwoBodyGate([locaNumberOp(PL), localIdentityOp(PR)]);
    IdNu = convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), locaNumberOp(PR)]);

    # compute μ and ν
    μ = cosh(ξ);
    ν = sinh(ξ);

    # construct K0, K1 and K2
    # K0 = -log(μ) * (NuId + IdNu + IdId);
    # K1 = conj(ν)/μ * AnAn;
    # K2 = -ν/μ * CrCr;
    K0 = -log(μ) * (NuId + IdNu + IdId);
    K1 = tanh(ξ) * AnAn;
    K2 = -tanh(ξ) * CrCr;

    # construct squeezing operator
    A = exp(K2);
    B = exp(K0);
    C = exp(K1);
    S = exp(K2) * exp(K0) * exp(K1);

    # construct derivative of squeezing operator
    dA = -1 * CrCr * (dν(ξ)/μ - ν * dμ(ξ)/μ^2);
    dB = -1 * (NuId + IdNu + IdId) * dμ(ξ)/μ;
    dC = +1 * AnAn * (dν(ξ)/μ - ν * dμ(ξ)/μ^2);
    # dSdXi = A * dA * B * C + A * B * dB * C + A * B * C * dC;
    # dSdXi = S * AnAn - CrCr * S;
    dSdXi = S * (AnAn - CrCr);
    return dSdXi;

end

function gradient_squeezed_tensor(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace, twoSiteTensor::TensorMap)
    dSdXi = gradient_squeezing_operator(ξ, nMax, kL, kR, PL, PR);
    dPsiXi = applyTwoModeTransformation(dSdXi, twoSiteTensor);
    return dPsiXi;
end

#---------------------------------------------------------------------------