#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using JLD
using LinearAlgebra: diagm
using HTTN
using LaTeXStrings
using Plots
using Printf
using TensorKit


# set truncation parameters
modelName = "sineGordon";
truncMethod = 3;
kMax = 2;
nMax = 3;
nMaxZM = 10;
modeOrdering = 1;
bogoliubovR = 0;

# truncMethod = 3;
# kMax = 2;
# nMax = 3;
# nMaxZM = 10;

# compute ground or excited state
computeEigenstate_0 = 1;
computeEigenstate_1 = 0;

# use numerical basis optimization
useBasisOptimization_0 = 0;
useBasisOptimization_1 = 0;

# # β = 1.0 * sqrt(4π)
# # truncMethod = 3
# # nMax = 2, nMaxZM = 20
# optimalBogParameters = [
#     [0.8826048683738424], 
#     [0.9828021791747763, 0.6491902344524971], 
#     [1.044626054859938, 0.7083610403948314, 0.5226475079764566], 
#     [1.0864793212772548, 0.747793190933278, 0.561024939327877, 0.4365535698356723], 
#     [1.1167481237210954, 0.7765059016219192, 0.5874806631594979, 0.46245394976334214, 0.3725019267150249], 
#     [1.1392177414024305, 0.7996105297696293, 0.6064367217222117, 0.48030611206315665, 0.39039125843460976, 0.32251603639744303], 
#     [1.1558472437952432, 0.8161322683819685, 0.6214373360865272, 0.4930055858509503, 0.4026399385554361, 0.33496731102922384, 0.28191646142112725], 
#     [1.168313945446149, 0.828516288649089, 0.6335752562582665, 0.4930055858509503, 0.4026399385554361, 0.33496731102922384, 0.2907932559537785, 0.24875574737005915], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [1.2154420098990029, 0.8741046214248941, 0.6850565405410879, 0.5555723860457121, 0.46204441091204673, 0.39140940910303207, 0.3343609068000002, 0.2870721376453025, 0.2506922636472108, 0.22253401068003634, 0.1973623119556844, 0.17389025874470296, 0.1556768567075738, 0.1402171933641977, 0.1285253169833322, 0.1162135859624253, 0.10552888433599983, 0.0968451306154898, 0.09113985045696667], 
#     [1.2225193774020136, 0.8803716914769047, 0.6879418108574467, 0.5577988583316977, 0.46376124715920797, 0.39367291936296134, 0.3376604157816094, 0.29218912736293856, 0.2539510721792752, 0.22253401068003634, 0.1973623119556844, 0.17715719756721635, 0.15848590999260423, 0.14259836529178319, 0.1285253169833322, 0.1162135859624253, 0.10820656279034487, 0.09925461079534144, 0.09113985045696667, 0.08391637833943519], 
# ];
# bogParameters = convert.(Float64, optimalBogParameters[kMax]);
# bogParameters = randn(Float64, kMax);

# # β = 1.0 * sqrt(4π), L = 15.0
# # truncMethod = 5
# # nMax = 10, nMaxZM = 10
# optimalBogParameters = [
#     [0.8835909604885277], 
#     [], 
#     [0.8710599494243699, 0.5467429252166217, 0.3804524031667206], 
#     [], 
#     [], 
#     [], 
#     [], 
#     [], 
# ];
# bogParameters = convert.(Float64, optimalBogParameters[kMax]);
# # bogParameters = randn(kMax);

# # truncMethod = 5
# # nMax = 10, nMaxZM = 10
# optimalBogParameters = [
#     [-0.8],
# ];
# bogParameters = convert.(Float64, optimalBogParameters[kMax]);
bogParameters = randn(kMax);

# set model parameters
βFF = 0.4 * sqrt(4 * π);
λ = 1.0;
L = 15.0;
# R = sqrt(4 * π) / β;

# set list of β
# betaList = βFF * collect(0.2 : 0.2 : 0.2);
betaList = βFF * [0.25];

# set list of L
sizeList = convert.(Float64, collect(1 : 1 : 25));
sizeList = [15.0];

# set DMRG parameters
bondDim = 1024;
truncErr = 1e-6;
subspaceExpansion = true;

# initialize or load previous simulation
initMethod = 0;
load_kMax = kMax;
load_modeOrdering = modeOrdering;
load_bogoliubovR = 0;
load_L = L;
load_bondDim = bondDim;

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

    for (idxL, L) in enumerate(sizeList)

        # create NamedTuple for truncation parameters and model parameters
        truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM, truncMethod = truncMethod, modeOrdering = modeOrdering, bogoliubovR = bogoliubovR, bogParameters = bogParameters);
        hamiltonianParameters = (β = β, R = R, λ = λ, L = L);

        # construct Sine-Gordon model (with MPO)
        sG = SineGordonModel(truncationParameters, hamiltonianParameters);
        display(sG.modeOccupations)

        # construct physical and virtual vector spaces for the MPS
        boundarySpaceL = U1Space(0 => 1);
        boundarySpaceR = U1Space(0 => 1);
        physSpaces = sG.physSpaces;
        virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);

        # initialize MPS
        if initMethod == 0
            
            # initialize vacuum MPS
            # initialMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
            vacuumMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
            initialMPS = initializeMPS(sG, vacuumMPS, modeOrdering = modeOrdering);
            # initialMPS = SparseMPS(randn, ComplexF64, physSpaces, virtSpaces);

            # set startOptimization after first DMRG sweep
            startOptimization = 1;

        elseif initMethod == 1

            # construct fileString
            loadFolderStringMPS = "numFiles/" * modelName * "/modeOrdering_" * string(load_modeOrdering) * "/bogoliubovRot_" * string(load_bogoliubovR) * "/truncMethod_" * string(truncMethod) * "/MPS/kMax_" * string(load_kMax);

            # construct fileString
            # fileStringMPS = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_L_%0.2f_beta_%0.3f_bondDim_%d.jld", loadFolderStringMPS, load_kMax, nMax, nMaxZM, load_L, β, load_bondDim);
            fileStringMPS = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_L_%0.2f_beta_%0.3f_bondDim_%d_0.jld", loadFolderStringMPS, load_kMax, nMax, nMaxZM, load_L, β, load_bondDim);
            println(fileStringMPS)

            # load groundStateMPS
            initialMPS = load(fileStringMPS, "groundStateMPS");
            initialMPS = SparseMPS(convert.(TensorMap, initialMPS));
            display(initialMPS[1])
    
            # # load groundStateMPS
            # initialMPS = load(fileStringMPS, "groundStateMPS");
            # initialMPS = SparseMPS(convert.(TensorMap, initialMPS[1 : (2 * kMax + 1)]));
            # projVirtSpaceR = TensorMap(ones, space(initialMPS[end], 3)', U1Space(0 => 1));
            # @tensor initialMPS[end][-1 -2; -3] := initialMPS[end][-1, -2, 3] * projVirtSpaceR[3, -3];

            # get optimal bogParameters for groundState and update sG model
            if load_bogoliubovR == 1
                storeBogoliubovParameters = load(fileStringMPS, "storeBogoliubovParameters");
                bogParameters = storeBogoliubovParameters[end, :];
                sG = updateBogoliubovParameters(sG, bogParameters);
                println(bogParameters)
            end

            # set startOptimization after first DMRG sweep
            startOptimization = 0;

        end


        # construct main directory path to store simulation files
        mainDirPath = "numFiles/" * modelName * "/modeOrdering_" * string(modeOrdering) * "/bogoliubovRot_" * string(bogoliubovR) * "/truncMethod_" * string(truncMethod);

        # construct fileString
        folderStringMPS = mainDirPath * "/MPS/kMax_" * string(kMax);
        ~isdir(folderStringMPS) && mkpath(folderStringMPS)

        # construct fileString
        fileStringMPS_0 = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_L_%0.2f_beta_%0.3f_bondDim_%d_0.jld", folderStringMPS, kMax, nMax, nMaxZM, L, β, bondDim);
        fileStringMPS_1 = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_L_%0.2f_beta_%0.3f_bondDim_%d_1.jld", folderStringMPS, kMax, nMax, nMaxZM, L, β, bondDim);


        # ground state
        if computeEigenstate_0 == 1

            storeBogoliubovParameters = [];
            if useBasisOptimization_0 == 0

                # construct sineGordon MPO
                hamMPO = generate_MPO_sG(sG);
                println(getLinkDimsMPO(hamMPO))

                # run DMRG for ground state
                groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = 1));
                @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

                # get MPS bond dimensions
                println(getLinkDimsMPS(groundStateMPS))

                # compute and plot entanglement entropy
                mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS);
                println(mpsEntanglementEntropies)
                
                labelString = @sprintf("\\beta = %0.1f \\cdot \\beta_{\\textrm{FF}}", β/βFF);
                labelString = latexstring(labelString);
                plot!(entanglementEntropyPlot, collect(0.5 : 1 : (2 * kMax)), mpsEntanglementEntropies, 
                    linewidth = 2.0, 
                    label = labelString, 
                );

                # store ground state MPS
                save(fileStringMPS_0, 
                    "sG", sG, 
                    "groundStateMPS", convert.(Dict, groundStateMPS),
                    "groundStateEnergy", groundStateEnergy, 
                    "truncErrors", truncErrors, 
                );
                
            elseif useBasisOptimization_0 == 1

                # run DMRG for ground state
                groundStateMPS, groundStateEnergy, storeBogoliubovParameters, truncErrors = find_groundstate(initialMPS, generate_MPO_sG, sG, DMRG2BO(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, startOptimization = startOptimization, verbosePrint = 2));
                @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

                # get MPS bond dimensions
                println(getLinkDimsMPS(groundStateMPS))

                # compute and plot entanglement entropy
                mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS);
                println(mpsEntanglementEntropies)
                
                labelString = @sprintf("\\beta = %0.1f \\cdot \\beta_{\\textrm{FF}}", β/βFF);
                labelString = latexstring(labelString);
                plot!(entanglementEntropyPlot, collect(0.5 : 1 : (2 * kMax)), mpsEntanglementEntropies, 
                    linewidth = 2.0, 
                    label = labelString, 
                );        

                # store ground state MPS
                save(fileStringMPS_0, 
                    "sG", sG, 
                    "groundStateMPS", convert.(Dict, groundStateMPS), 
                    "groundStateEnergy", groundStateEnergy, 
                    "truncErrors", truncErrors, 
                    "storeBogoliubovParameters", storeBogoliubovParameters
                );

            end

        end

        # first excited state
        if computeEigenstate_1 == 1

            if useBasisOptimization_1 == 0

                # get optimal bogParameters for groundState and update sG model
                # bogParameters = storeBogoliubovParameters[end, :];
                sG = updateBogoliubovParameters(sG, bogParameters);
                # println(bogParameters)

                # construct sineGordon MPO
                hamMPO = generate_MPO_sG(sG);
                println(getLinkDimsMPO(hamMPO))

                # run DMRG for excited state
                lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
                excitedStateMPS, excitedStateEnergy = find_excitedstate(initialMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = bondDim, truncErr = truncErr, verbosePrint = true));
                @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

                # get MPS bond dimensions
                println(getLinkDimsMPS(excitedStateMPS))

                # compute entanglement entropy
                mpsEntanglementEntropies = compute_entanglement_entropies(excitedStateMPS);
                println(mpsEntanglementEntropies)

                labelString = @sprintf("\\beta = %0.1f \\cdot \\beta_{\\textrm{FF}}", β/βFF);
                labelString = latexstring(labelString);
                plot!(entanglementEntropyPlot, collect(0.5 : 1 : (2 * kMax)), mpsEntanglementEntropies, 
                    linewidth = 2.0, 
                    label = labelString, 
                );

                # store excited state MPS
                save(fileStringMPS_1, 
                    "sG", sG, 
                    "excitedStateMPS", convert.(Dict, excitedStateMPS), 
                    "excitedStateEnergy", excitedStateEnergy, 
                    "truncErrors", truncErrors, 
                );

            elseif useBasisOptimization_1 == 1

                # run DMRG for excited state
                lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
                excitedStateMPS, excitedStateEnergy, storeBogoliubovParameters = find_excitedstate(groundStateMPS, generate_MPO_sG, lowEnergyStates, sG, DMRG2BO(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = false, startOptimization = startOptimization, verbosePrint = 1));
                @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

                # get MPS bond dimensions
                println(getLinkDimsMPS(excitedStateMPS))

                # compute entanglement entropy
                mpsEntanglementEntropies = compute_entanglement_entropies(excitedStateMPS);
                println(mpsEntanglementEntropies)

                labelString = @sprintf("\\beta = %0.1f \\cdot \\beta_{\\textrm{FF}}", β/βFF);
                labelString = latexstring(labelString);
                plot!(entanglementEntropyPlot, collect(0.5 : 1 : (2 * kMax)), mpsEntanglementEntropies, 
                    linewidth = 2.0, 
                    label = labelString, 
                );

                # store excited state MPS
                save(fileStringMPS_1, 
                    "sG", sG, 
                    "excitedStateMPS", convert.(Dict, excitedStateMPS), 
                    "excitedStateEnergy", excitedStateEnergy, 
                    "truncErrors", truncErrors, 
                    "storeBogoliubovParameters", storeBogoliubovParameters
                );

            end

            @printf("energy gap Δ = %0.8f\n", excitedStateEnergy - groundStateEnergy)

        end

        # # get MPS linkDims
        # println(getLinkDimsMPS(groundStateMPS))
        # println(getLinkDimsMPS(excitedStateMPS))

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
        #     sG = updateBogoliubovParameters(sG, finalBogParameters);
        #     hamMPO = generate_MPO_sG(sG);

        #     # # run DMRG for excited state
        #     # lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
        #     # excitedStateMPS, excitedStateEnergy = find_excitedstate(groundStateMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
        #     # @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

        # end

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


        # # compute mutual information between all pairs of modes [k, k']
        # momentumModes = getMomentumModes(sG);
        # entanglementEntropyMatrix = compute_mutual_information(getMomentumModes(sG), excitedStateMPS);
        # display(entanglementEntropyMatrix)
        # entanglementHeatMap = heatmap(collect(0 : (2 * kMax)), collect(0 : (2 * kMax)), entanglementEntropyMatrix, 
        #     xticks = xTicks, 
        #     yticks = xTicks, 
        #     lims = (0, 2 * kMax), 
        #     c = cgrad(:Blues), 
        #     aspect_ratio = :equal
        # );
        # display(entanglementHeatMap)


        # # compute mutual information between zero mode and pairs of modes [0, ±k]
        # momentumModes = getMomentumModes(sG);
        # mutualInformationPairs = compute_mutual_information_pairs(getMomentumModes(sG), groundStateMPS);
        # display(mutualInformationPairs)

        # xTicks = ((collect(0 : kMax)), [kVal == 0 ? latexstring(@sprintf("%d", kVal)) : latexstring(@sprintf("\\pm %d", kVal)) for kVal in collect(0 : kMax)]);
        # mutualInformationPlot = plot(collect(0 : kMax), mutualInformationPairs, 
        #     xticks = xTicks, 
        #     linewidth = 2.0, 
        #     xlab = "pair of modes", 
        #     ylab = L"\mathcal{I}(0, \pm k)", 
        #     label = "", 
        #     frame = :box, 
        # );
        # display(mutualInformationPlot)


        # # compute mutual information between zero mode and pairs of modes [0, ±k]
        # momentumModes = getMomentumModes(sG);
        # mutualInformationPairs = compute_mutual_information_pairs(getMomentumModes(sG), excitedStateMPS);
        # display(mutualInformationPairs)

        # xTicks = ((collect(0 : kMax)), [kVal == 0 ? latexstring(@sprintf("%d", kVal)) : latexstring(@sprintf("\\pm %d", kVal)) for kVal in collect(0 : kMax)]);
        # mutualInformationPlot = plot(collect(0 : kMax), mutualInformationPairs, 
        #     xticks = xTicks, 
        #     linewidth = 2.0, 
        #     xlab = "pair of modes", 
        #     ylab = L"\mathcal{I}(0, \pm k)", 
        #     label = "", 
        #     frame = :box, 
        # );
        # display(mutualInformationPlot)

    end

end
# display(entanglementEntropyPlot)

# # compute local occupation numbers
# numberOperators = local_number_operators(sG);
# localOccupations = expectation_values(groundStateMPS, numberOperators);
# println(localOccupations)

# localOccupations = expectation_values(excitedStateMPS, numberOperators);
# println(localOccupations)


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