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
    return tr(S);
end

function computeEntropy(twoSiteTensor::TensorMap)
    _, S, _ = tsvd(twoSiteTensor, (1, 2), (3, 4));
    vnEntropy = abs(-tr(S^2 * log(S^2)));
    return vnEntropy;
end

function matrixExponentialSeries(operator::TensorMap, nMax::Int64)
    matrixExp = sum([1/factorial(n) * operator^n for n in collect(0 : nMax)]);
    return matrixExp;
end

# construct squeezing operator
function squeezingOp(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace)

    # construct two-site operators
    CrCr = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localCreationOp(kL, PL), localCreationOp(kR, PR)]);
    AnAn = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localAnnihilationOp(kL, PL), localAnnihilationOp(kR, PR)]);
    IdId = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), localIdentityOp(PR)]);
    NuId = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([locaNumberOp(PL), localIdentityOp(PR)]);
    IdNu = @Zygote.ignore convertLocalOperatorsToTwoBodyGate([localIdentityOp(PL), locaNumberOp(PR)]);

    # compute μ and ν
    μ = cosh(abs(ξ));
    ν = exp(1im * angle(ξ)) * sinh(abs(ξ));

    # construct K0, K1 and K2
    K0 = -log(μ) * (NuId + IdNu + IdId);
    K1 = conj(ν)/μ * AnAn;
    K2 = -ν/μ * CrCr;

    # construct squeezing operator
    S = matrixExponentialSeries(K2, nMax) * matrixExponentialSeries(K0, nMax) * matrixExponentialSeries(K1, nMax);
    # S = exp(K2) * exp(K0) * exp(K1);
    return S;

end

function findDisentanglingRotation(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace, twoSiteTensor::TensorMap)
    twoSiteSqueezingOperator = squeezingOp(ξ, nMax, kL, kR, PL, PR);
    costFunction = computeRenyiEntropy(applyTwoModeTransformation(twoSiteSqueezingOperator, twoSiteTensor));
    return costFunction;
end

function value_and_gradient(ξ::Union{Int64, Float64, ComplexF64}, nMax::Int64, kL::Int64, kR::Int64, PL::ElementarySpace, PR::ElementarySpace, twoSiteTensor::TensorMap)
    fval = findDisentanglingRotation(ξ, nMax, kL, kR, PL, PR, twoSiteTensor);
    gval = Zygote.gradient(x -> findDisentanglingRotation(x, nMax, kL, kR, PL, PR, twoSiteTensor), ξ)[1];
    return real(fval), gval;
end

# OptimKit functions for optimization of Float64 values
_scale!(v, α) = v * α;
_add!(vdst, vsrc, α) = vdst += vsrc * α;