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
bogParameters = [1.24, 0.90, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13];
bogParameters = bogParameters[1 : kMax];

# bogParameterInitial(kMax::Union{Int64, Float64}) = 1.5882 - 0.5014 * kMax + 0.0727 * kMax^2 - 0.0038 * kMax^3 + 0.05 * rand();
# bogParameters = bogParameterInitial.(collect(1 : kMax));


# set model parameters
βFF = sqrt(4 * π);
λ = 1.0;
L = 25.0;
# R = sqrt(4 * π) / β;

# set list of betas
betaList = βFF * collect(1.0 : 0.2 : 1.0);

# set DMRG parameters
bondDim = 2048;
truncErr = 1e-6;
subspaceExpansion = false;

# use numerical basis optimization
useBasisOptimization = bogoliubovR;

# initialize or load previous simulation
initMethod = 0;
loadModeOrdering = 0;
loadBogoliubovR = 0;
loadL = 15.0;
loadBondDim = 2000;

# plot entanglement entropy
xLimits = (0, 2 * kMax);
momentumModes = convert.(Int64, collect(-kMax : 1 : +kMax));
if modeOrdering == 1
    momentumModes = sort(momentumModes, by = abs);
end
xTicks = (
    (collect(0 : (2 * kMax))), [kVal == 0 ? latexstring(@sprintf("%d", kVal)) : latexstring(@sprintf("%+d", kVal)) for kVal in momentumModes]
)

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


mpsEntanglementEntropies = [];
entanglementEntropyMatrix = [];

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

    # initialize MPS
    if initMethod == 0
        vacuumMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(sG, vacuumMPS, modeOrdering = modeOrdering);
        # initialMPS = SparseMPS(randn, ComplexF64, physSpaces, virtSpaces);

        # set startOptimization after first DMRG sweep
        startOptimization = 1;

    elseif initMethod == 1

        # construct fileString
        loadFolderStringMPS = "numFiles/" * modelName * "/modeOrdering_" * string(loadModeOrdering) * "/bogoliubovRot_" * string(loadBogoliubovR) * "/truncMethod_" * string(truncMethod) * "/MPS/kMax_" * string(kMax);

        # construct fileString
        fileStringMPS = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_L_%0.2f_beta_%0.3f_bondDim_%d.jld", loadFolderStringMPS, kMax, nMax, nMaxZM, loadL, β, loadBondDim);
        # fileStringMPS = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_L_%0.2f_beta_%0.3f_bondDim_%d_0.jld", loadFolderStringMPS, kMax, nMax, nMaxZM, loadL, β, loadBondDim);
        println(fileStringMPS)

        # load groundStateMPS
        # initialMPS = load(fileStringMPS, "groundStateMPS");
        initialMPS = load(fileStringMPS, "groundState_fM");
        initialMPS = SparseMPS(convert.(TensorMap, initialMPS));

        # set startOptimization after first DMRG sweep
        startOptimization = 0;

    end


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

    # run DMRG for ground state
    groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = true));
    @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

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

    # run DMRG for ground state
    groundStateMPS, groundStateEnergy, storeBogoliubovParameters, truncErrors = find_groundstate(initialMPS, generate_MPO_sG, sG, DMRG2BO(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, startOptimization = startOptimization, verbosePrint = true));
    @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

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

    # get MPS linkDims
    println(getLinkDimsMPS(groundStateMPS))

    # construct main directory path to store simulation files
    mainDirPath = "numFiles/" * modelName * "/modeOrdering_" * string(modeOrdering) * "/bogoliubovRot_" * string(bogoliubovR) * "/truncMethod_" * string(truncMethod);

    # construct fileString
    folderStringMPS = mainDirPath * "/MPS/kMax_" * string(kMax);
    ~isdir(folderStringMPS) && mkpath(folderStringMPS)

    # construct fileString
    fileStringMPS = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_L_%0.2f_beta_%0.3f_bondDim_%d_0.jld", folderStringMPS, kMax, nMax, nMaxZM, L, β, bondDim);

    # store MPS states and energies
    if useBasisOptimization == 0
        save(fileStringMPS, 
            "sG", sG, 
            "groundStateMPS", convert.(Dict, groundStateMPS), 
            "truncErrors", truncErrors, 
            "groundStateEnergy", groundStateEnergy
        );
    elseif useBasisOptimization == 1
        save(fileStringMPS, 
            "sG", sG, 
            "groundStateMPS", convert.(Dict, groundStateMPS), 
            "groundStateEnergy", groundStateEnergy, 
            "truncErrors", truncErrors, 
            "storeBogoliubovParameters", storeBogoliubovParameters
        );
    end
    display(fileStringMPS)


    # compute entanglement entropies
    mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS);
    println(mpsEntanglementEntropies)

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


    # # compute mutual information between all pairs of modes [k, k']
    # momentumModes = getMomentumModes(sG);
    # entanglementEntropyMatrix = compute_mutual_information(getMomentumModes(sG), groundStateMPS);
    # display(entanglementEntropyMatrix)
    # entanglementHeatMap = heatmap(collect(0 : (2 * kMax)), collect(0 : (2 * kMax)), entanglementEntropyMatrix, 
    #     xticks = xTicks, 
    #     yticks = xTicks, 
    #     lims = (0, 2 * kMax), 
    #     c = cgrad(:Blues), 
    #     aspect_ratio = :equal
    # );
    # display(entanglementHeatMap)


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