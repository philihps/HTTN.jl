@kwdef struct TDVP2
    bondDim::Int64 = 10
    truncErrT::Float64 = 1e-6
    verbosePrint = false
end

function perform_timestep!(finiteMPS::SparseMPS, finiteMPO::SparseMPO, timeStep::Union{Float64, ComplexF64}, alg::TDVP2)
    """ 2-site TDVP implementation for finiteMPO with global Krylov subspace expansion """

    # initialize MPO environments
    mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO);

    # initialize truncationErrors
    truncationErrors = Float64[];

    # sweep L ---> R
    for siteIdx = 1 : +1 : (length(finiteMPS) - 1)

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        H_AC2 = ∂∂AC2(siteIdx, finiteMPO, mpoEnvL, mpoEnvR);
        newAC2, convHist = exponentiate(H_AC2, -1im * timeStep / 2, AC2, Lanczos());

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT), alg = TensorKit.SVD());
        S /= norm(S);
        U = permute(U, (1, 2), (3, ));
        V = permute(S * V, (1, 2), (3, ));
        truncationErrors = vcat(truncationErrors, ϵ);

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U;
        finiteMPS[siteIdx + 1] = V;

        # update mpoEnvL
        mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

        # compute K(n + 1) and apply it to V(n + 1) to evolve it with exp(+1im * timeStep/2 * K(n + 1))
        if siteIdx < (length(finiteMPS) - 1)
            H_AC = ∂∂AC(siteIdx + 1, finiteMPO, mpoEnvL, mpoEnvR);
            finiteMPS[siteIdx + 1], convHist = exponentiate(H_AC, +1im * timeStep / 2, finiteMPS[siteIdx + 1], Lanczos());
        end

    end

    # sweep L <--- R
    for siteIdx = (length(finiteMPS) - 1) : -1 : 1

        # construct initial AC
        AC2 = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

        # compute H(n, n + 1) and apply it to AC(n, n + 1) to evolve it with exp(-1im * timeStep/2 * H(n, n + 1))
        H_AC2 = ∂∂AC2(siteIdx, finiteMPO, mpoEnvL, mpoEnvR);
        newAC2, convHist = exponentiate(H_AC2, -1im * timeStep / 2, AC2, Lanczos());

        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(newAC2, (1, 2), (3, 4), trunc = truncdim(alg.bondDim) & truncerr(alg.truncErrT), alg = TensorKit.SVD());
        S /= norm(S);
        U = permute(U * S, (1, 2), (3, ));
        V = permute(V, (1, 2), (3, ));
        truncationErrors = vcat(truncationErrors, ϵ);

        # assign updated tensors
        finiteMPS[siteIdx + 0] = U;
        finiteMPS[siteIdx + 1] = V;

        # update mpoEnvR
        mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1], finiteMPS[siteIdx + 1], finiteMPO[siteIdx + 1], finiteMPS[siteIdx + 1]);

        # compute K(n) and apply it to V(n) to evolve it with exp(+1im * timeStep/2 * K(n))
        if siteIdx > 1
            H_AC = ∂∂AC(siteIdx + 0, finiteMPO, mpoEnvL, mpoEnvR);
            finiteMPS[siteIdx + 0], convHist = exponentiate(H_AC, +1im * timeStep / 2, finiteMPS[siteIdx + 0], Lanczos());
        end

    end

    # return optimized finiteMPS
    return finiteMPS, mpoEnvL, mpoEnvR, maximum(truncationErrors);

end

function perform_timestep(finiteMPS::SparseMPS, finiteMPO::SparseMPO, timeStep::Union{Float64, ComplexF64}, alg::TDVP2)
    return perform_timestep!(copy(finiteMPS), finiteMPO, timeStep, alg)
end

#-------------------------------------------------
# code copied and modified from MPSKit
# https://github.com/maartenvd/MPSKit.jl

struct MPO_∂∂C
    envL::TensorMap
    envR::TensorMap
end
∂∂C(siteIdx::Int, envL::Vector{<:AbstractTensorMap}, envR::Vector{<:AbstractTensorMap}) = 
    MPO_∂∂C(envL[siteIdx + 1], envR[siteIdx + 0]);

struct MPO_∂∂AC
    MPO::TensorMap
    envL::TensorMap
    envR::TensorMap
end
∂∂AC(siteIdx::Int, MPO::SparseMPO, envL::Vector{<:AbstractTensorMap}, envR::Vector{<:AbstractTensorMap}) = 
    MPO_∂∂AC(MPO[siteIdx + 0], envL[siteIdx + 0], envR[siteIdx + 0]);

struct MPO_∂∂AC2
    MPOL::TensorMap
    MPOR::TensorMap
    envL::TensorMap
    envR::TensorMap
end
∂∂AC2(siteIdx::Int, MPO::SparseMPO, envL::Vector{<:AbstractTensorMap}, envR::Vector{<:AbstractTensorMap}) = 
    MPO_∂∂AC2(MPO[siteIdx + 0], MPO[siteIdx + 1], envL[siteIdx + 0], envR[siteIdx + 1]);

Base.:*(h::Union{<:MPO_∂∂C, <:MPO_∂∂AC, <:MPO_∂∂AC2}, v) = h(v);
(h::MPO_∂∂C)(x) = ∂C(x, h.envL, h.envR);
(h::MPO_∂∂AC)(x) = ∂AC(x, h.MPO, h.envL, h.envR);
(h::MPO_∂∂AC2)(x) = ∂AC2(x, h.MPOL, h.MPOR, h.envL, h.envR);

function ∂C(x::TensorMap, envL::TensorMap, envR::TensorMap)
    @tensor x[-1; -2] := envL[-1, 3, 1] * x[1, 2] * envR[2, 3, -2];
    return x;
end

function ∂AC(x::TensorMap, mpo::TensorMap, envL::TensorMap, envR::TensorMap)
    @tensor x[-1 -2; -3] := envL[-1, 2, 1] * x[1, 3, 4] * mpo[2, -2, 5, 3] * envR[4, 5, -3]
    return x;
end

function ∂AC2(x::TensorMap, mpoL::TensorMap, mpoR::TensorMap, envL::TensorMap, envR::TensorMap)
    @tensor x[-1 -2; -3 -4] := envL[-1, 2, 1] * x[1, 3, 5, 6] * mpoL[2, -2, 4, 3] * mpoR[4, -3, 7, 5] * envR[6, 7, -4];
end