#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using LinearAlgebra: diagm
using HTTN
using LaTeXStrings
using Plots
using Printf
using TensorKit

include("/Users/philipp/Downloads/MPOcompression.jl")


# set truncation parameters
truncMethod = 5;
kMax = 6;
nMax = 8;
nMaxZM = 12;
modeOrdering = 1;
bogoliubovR = 0;
bogParameters = 0.4 .+ 0.1 * reverse(collect(1 : kMax));
# bogParameters = [1.25, 0.7658787726053576, 0.5916149758233501, 0.46691233917615466, 0.37809892346039614, 0.31132850050333677];
bogParameters = [1.0730311449195271, 0.7349048939229015, 0.5492985081346116, 0.42663356984035494, 0.3, 0.2];
bogParameters = [ 1.13, 0.80, 0.61, 0.48, 0.39, 0.32, 0.27, 0.22];
# bogParameters = [1.07, 0.64, 0.55, 0.43];
# bogParameters = [0.974, 0.642, 0.4];
# bogParameters = [0.974, 0.642];
bogParameters = rand(kMax);

# set model parameters
βFF = sqrt(4 * π);
λ = 1.0;
L = 25.0;
# R = sqrt(4 * π) / β;

# set list of betas
betaList = βFF * collect(0.2 : 0.2 : 1.2);

# set DMRG parameters
bondDim = 2500;
truncErr = 1e-4;

# use numerical basis optimization
useBasisOptimization = bogoliubovR;

# plot entanglement entropy
xLimits = (0, 2 * kMax);
if kMax == 3
    xTicks = (collect(0 : (2 * kMax)), (L"0", L"-1", L"+1", L"-2", L"+2", L"-3", L"+3"));
elseif kMax == 4
    xTicks = (collect(0 : (2 * kMax)), (L"0", L"-1", L"+1", L"-2", L"+2", L"-3", L"+3", L"-4", L"+4"));
elseif kMax == 6
    xTicks = (collect(0 : (2 * kMax)), (L"0", L"-1", L"+1", L"-2", L"+2", L"-3", L"+3", L"-4", L"+4", L"-5", L"+5", L"-6", L"+6"));
end

# initialize plot for entanglement entropy
entanglementEntropyPlot = plot( 
    xlims = xLimits, 
    xticks = xTicks, 
    xlab = L"k", 
    ylims = (0, Inf), 
    ylab = L"S(k_L,k_R)", 
    linewidth = 2.0, 
    legend = :topright, 
    frame = :box
);

# loop over Hamiltonian parameters
for (idxB, β) in enumerate(betaList)

    # compute compactification radius
    R = sqrt(4 * π) / β;

    # create NamedTuple for truncation parameters and model parameters
    truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM, truncMethod = truncMethod, modeOrdering = modeOrdering, bogoliubovR = bogoliubovR, bogParameters = bogParameters);
    hamiltonianParameters = (β = β, R = R, λ = λ, L = L);

    # construct Sine-Gordon model (with MPO)
    sG = SineGordonModel(truncationParameters, hamiltonianParameters);
    display(sG.modeOccupations)

    # construct sineGordon MPO
    hamMPO = generate_MPO_sG(sG);
    println(getLinkDimsMPO(hamMPO))

    # # compress MPO
    # nonCompressedMPO = Vector{TensorMap}(undef, length(hamMPO));
    # for idx = eachindex(hamMPO)
    #     nonCompressedMPO[idx] = hamMPO[idx];
    # end
    # compressedMPO = compress_MPO(nonCompressedMPO)

    # println("Comparison of the bond dimensions")
    # for j in 1:length(compressedMPO)
    #     mat = nonCompressedMPO[j]
    #     println("Position $j")
    #     println("Original size: $((dim(mat.codom[2]), dim(mat.dom[1])) )")
    #     mat = compressedMPO[j]
    #     println("After compression: $((dim(mat.codom[2]), dim(mat.dom[1])) )")
    # end

    # cM = compressedMPO[1]; uCM = nonCompressedMPO[1];
    # localDim = dim(cM.codom[1])
    # @tensor temp[1; 2] := conj(cM[-1 -2; 1 -3]) * uCM[-1 -2; 2 -3] / localDim
    # for j in 2:length(compressedMPO)
    #     cM = compressedMPO[j]; uCM = nonCompressedMPO[j];
    #     localDim = dim(cM.codom[1])
    #     @tensor temp[1; 2] := temp[-1; -2] * conj(cM[-3 -1; 1 -4]) * uCM[-3 -2; 2 -4] / localDim
    # end
    # display("( comMPO, nonCompMPO ) : $temp")

    # cM = nonCompressedMPO[1]; uCM = nonCompressedMPO[1];
    # localDim = dim(cM.codom[1])
    # @tensor temp[1; 2] := conj(cM[-1 -2; 1 -3]) * uCM[-1 -2; 2 -3] / localDim
    # for j in 2:length(compressedMPO)
    #     cM = nonCompressedMPO[j]; uCM = nonCompressedMPO[j];
    #     localDim = dim(cM.codom[1])
    #     @tensor temp[1; 2] := temp[-1; -2] * conj(cM[-3 -1; 1 -4]) * uCM[-3 -2; 2 -4] / localDim
    # end
    # display("( nonCompMPO, nonCompMPO ) : $temp")

    # cM = compressedMPO[1]; uCM = compressedMPO[1];
    # localDim = dim(cM.codom[1])
    # @tensor temp[1; 2] := conj(cM[-1 -2; 1 -3]) * uCM[-1 -2; 2 -3] / localDim
    # for j in 2:length(compressedMPO)
    #     cM = compressedMPO[j]; uCM = compressedMPO[j];
    #     localDim = dim(cM.codom[1])
    #     @tensor temp[1; 2] := temp[-1; -2] * conj(cM[-3 -1; 1 -4]) * uCM[-3 -2; 2 -4] / localDim
    # end
    # display("( comMPO, comMPO ) : $temp")

    # # construct physical and virtual vector spaces for the MPS
    # boundarySpaceL = U1Space(0 => 1);
    # boundarySpaceR = U1Space(0 => 1);
    # physSpaces = sG.physSpaces;
    # virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);

    # # initialize MPS
    # initialMPS = SparseMPS(randn, ComplexF64, physSpaces, virtSpaces);

    # storeBogoliubovParameters = [];
    # if useBasisOptimization == 0

    #     # construct sineGordon MPO
    #     hamMPO = generate_MPO_sG(sG);
    #     println(getLinkDimsMPO(hamMPO))

    #     # run DMRG for ground state
    #     groundStateMPS, groundStateEnergy = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, verbosePrint = true));
    #     @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

    #     # # run DMRG for excited state
    #     # lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
    #     # excitedStateMPS, excitedStateEnergy = find_excitedstate(groundStateMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = bondDim, truncErr = truncErr, verbosePrint = true));
    #     # @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

    # elseif useBasisOptimization == 1

    #     # run DMRG for ground state
    #     groundStateMPS, groundStateEnergy, storeBogoliubovParameters = find_groundstate(initialMPS, generate_MPO_sG, sG, DMRG2BO(bondDim = bondDim, truncErr = truncErr, verbosePrint = true));
    #     @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

    #     # # plot bogParameters
    #     # bogParametersPlot = plot(frame = :box);
    #     # for kIdx = axes(storeBogoliubovParameters, 2)
    #     #     labelString = @sprintf("k = \\pm %d", kIdx);
    #     #     labelString = latexstring(labelString);
    #     #     plot!(bogParametersPlot, collect(1 : size(storeBogoliubovParameters, 1)), storeBogoliubovParameters[:, kIdx], 
    #     #         markers = :circle, 
    #     #         linewidth = 2.0,
    #     #         label = labelString
    #     #     )
    #     # end
    #     # display(bogParametersPlot)

    #     # display(plot(1 : kMax, storeBogoliubovParameters[end, :], linewidth = 2.0, frame = :box))

    #     # construct rotated sineGordon MPO
    #     finalBogParameters = storeBogoliubovParameters[end, :];
    #     sG = updateBogoliubovPrameters(sG, finalBogParameters);
    #     hamMPO = generate_MPO_sG(sG);

    #     # # run DMRG for excited state
    #     # lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
    #     # excitedStateMPS, excitedStateEnergy = find_excitedstate(groundStateMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
    #     # @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

    # end

    # # compute entanglement entropies
    # mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS);
    # println(mpsEntanglementEntropies)

    # # mpsEntanglementEntropies = compute_entanglement_entropies(excitedStateMPS);
    # # println(mpsEntanglementEntropies)


    # # plot entanglement entropy
    # labelString = @sprintf("\\beta = %0.1f \\cdot \\beta_{\\textrm{FF}}", β/βFF);
    # labelString = latexstring(labelString);
    # plot!(entanglementEntropyPlot, collect(0.5 : 1 : (2 * kMax)), mpsEntanglementEntropies, 
    #     linewidth = 2.0, 
    #     label = labelString, 
    # );

end
# display(entanglementEntropyPlot)

# # compute local occupation numbers
# numberOperators = local_number_operators(sG);
# localOccupations = expectation_values(groundStateMPS, numberOperators);
# println(localOccupations)

# # localOccupations = expectation_values(excitedStateMPS, numberOperators);
# # println(localOccupations)


# # compute ⟨a(-k) a(+k)⟩
# mpos_AnAn, mpos_CrCr = pairing_operators(sG);
# expVals_AnAn = zeros(ComplexF64, kMax);
# expVals_CrCr = zeros(ComplexF64, kMax);
# for idx = 1 : kMax
#     expVals_AnAn[idx] = expectation_value_mpo(groundStateMPS, mpos_AnAn[idx]);
#     expVals_CrCr[idx] = expectation_value_mpo(groundStateMPS, mpos_CrCr[idx]);
# end
# println(real.(expVals_AnAn))
# # println(real.(expVals_CrCr))

# # expVals_AnAn = zeros(ComplexF64, kMax);
# # expVals_CrCr = zeros(ComplexF64, kMax);
# # for idx = 1 : kMax
# #     expVals_AnAn[idx] = expectation_value_mpo(excitedStateMPS, mpos_AnAn[idx]);
# #     expVals_CrCr[idx] = expectation_value_mpo(excitedStateMPS, mpos_CrCr[idx]);
# # end
# # println(real.(expVals_AnAn))
# # # println(real.(expVals_CrCr))


# # # # compute expectation value of vertex operator
# # # vertexOperatorExpVal = expectation_value_mpo(groundStateMPS, sG.mpo_H1);

# # get MPS linkDims
# println(getLinkDimsMPS(groundStateMPS))
# # println(getLinkDimsMPS(excitedStateMPS))