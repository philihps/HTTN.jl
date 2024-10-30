#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using DelimitedFiles
using HTTN
using JLD
using KrylovKit
using LaTeXStrings
using Plots
using Printf
using TensorKit


# set truncation parameters
modelName = "massiveSchwinger";
truncMethod = 5;
kMax = 2;
nMax = 10;
nMaxZM = 20;
modeOrdering = 1;
bogoliubovR = 0;
useBasisOptimization = 0;

# # truncMethod = 5, nMax = 10
# bogParameters = [0.21, 0.18];
# # bogParameters = [-0.3384];
# # bogParameters = bogParameters[1 : kMax];
# bogParameters = rand(kMax);
# bogParameters = [0.00];

# m = 0.1, L = 100.0
# truncMethod = 5
# nMax = 10, nMaxZM = 20
optimalBogParameters = [
    [0.21300345504641333], 
    [0.21300345504641333, 0.16], 
];
bogParameters = convert.(Float64, optimalBogParameters[kMax]);
# bogParameters = randn(kMax);

# set model parameters
θ = 1.0 * π;
e = 1.0;
M = e / sqrt(π);
L = 100.0;

# set fermion mass
fermionMasses = collect(0.100 : 0.10 : 0.100);
fermionMasses = [0.1];
# fermionMasses = collect(0.20 : 0.05 : 0.35);

# set DMRG parameters
bondDim = 1024;
truncErr = 1e-6;
subspaceExpansion = true;

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


# loop over Hamiltonian parameters
for (idxM, m) in enumerate(fermionMasses)


    # create NamedTuple for truncation parameters and model parameters
    truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM, truncMethod = truncMethod, modeOrdering = modeOrdering, bogoliubovR = bogoliubovR, bogParameters = bogParameters);
    hamiltonianParameters = (θ = θ, m = m, M = M, L = L);

    # construct Sine-Gordon model (with MPO)
    mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters);
    display(mS.modeOccupations)

    # construct physical and virtual vector spaces for the MPS
    boundarySpaceL = U1Space(0 => 1);
    boundarySpaceR = U1Space(0 => 1);
    physSpaces = mS.physSpaces;
    virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);

    storeBogoliubovParameters = [];
    if useBasisOptimization == 0

        # construct massiveSchwinger MPO
        hamMPO = generate_MPO_mS(mS);
        println(getLinkDimsMPO(hamMPO))
        # display(hamMPO)

        
        # initialize ground state
        # initialMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(mS, vacuumMPS, modeOrdering = modeOrdering);

        # run DMRG for ground state
        groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = 1));
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)


        # initialize excited state
        # initialMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(mS, vacuumMPS, modeOrdering = modeOrdering);

        # run DMRG for excited state
        lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
        excitedStateMPS, excitedStateEnergy = find_excitedstate(initialMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = true));
        @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

    elseif useBasisOptimization == 1

        # initialize ground state
        # initialMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(mS, vacuumMPS, modeOrdering = modeOrdering);

        # run DMRG for ground state
        groundStateMPS, groundStateEnergy, storeBogoliubovParameters, truncErrors = find_groundstate(initialMPS, generate_MPO_mS, mS, DMRG2BO(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion,  verbosePrint = 2));
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

        # # plot bogParameters
        # bogParametersPlot = plot(frame = :box);
        # for kIdx = axes(storeBogoliubovParameters, 2)
        #     labelString = @sprintf("k = \\pm %d", kIdx);
        #     labelString = latexstring(labelString);
        #     plot!(bogParametersPlot, collect(1 : size(storeBogoliubovParameters, 1)), storeBogoliubovParameters[:, kIdx], 
        #         markers = :circle, 
        #         linewidth = 2.0,
        #         label = labelString
        #     )
        # end
        # display(bogParametersPlot)

        # display(plot(1 : kMax, storeBogoliubovParameters[end, :], linewidth = 2.0, frame = :box))

        # construct rotated sineGordon MPO
        finalBogParameters = storeBogoliubovParameters[end, :];
        mS = updateBogoliubovParameters(mS, finalBogParameters);
        hamMPO = generate_MPO_mS(mS);

        # # run DMRG for excited state
        # lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
        # excitedStateMPS, excitedStateEnergy = find_excitedstate(groundStateMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
        # @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

    end

    # get MPS linkDims
    println(getLinkDimsMPS(groundStateMPS))
    # println(getLinkDimsMPS(excitedStateMPS))

    # construct main directory path to store simulation files
    mainDirPath = "numFiles/" * modelName * "/modeOrdering_" * string(modeOrdering) * "/bogoliubovRot_" * string(bogoliubovR) * "/truncMethod_" * string(truncMethod);

    # construct fileString
    folderStringMPS = mainDirPath * "/MPS/kMax_" * string(kMax);
    ~isdir(folderStringMPS) && mkpath(folderStringMPS)

    # construct fileString
    fileStringMPS = @sprintf("%s/mps_kMax_%d_nMax_%d_nMaxZM_%d_T_%0.2f_M_%0.2f_m_%0.3f_L_%0.2f_bondDim_%d.jld", folderStringMPS, kMax, nMax, nMaxZM, θ, M, m, L, bondDim);

    # store MPS states and energies
    if useBasisOptimization == 0
        save(fileStringMPS, 
            "mS", mS, 
            "groundStateMPS", convert.(Dict, groundStateMPS), 
            "truncErrors", truncErrors, 
            "groundStateEnergy", groundStateEnergy
        );
    elseif useBasisOptimization == 1
        save(fileStringMPS, 
            "mS", mS, 
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

    # mpsEntanglementEntropies = compute_entanglement_entropies(excitedStateMPS);
    # println(mpsEntanglementEntropies)


    # plot entanglement entropy
    labelString = @sprintf("m = %0.3f", m);
    labelString = latexstring(labelString);
    plot!(entanglementEntropyPlot, collect(0.5 : 1 : (2 * kMax)), mpsEntanglementEntropies, 
        linewidth = 2.0, 
        label = labelString, 
    );


    # # compute mutual information between all pairs of modes [k, k']
    # momentumModes = getMomentumModes(mS);
    # mutualInformationMatrix = compute_mutual_information(momentumModes, groundStateMPS);
    # mutualInformationMatrix ./= maximum(mutualInformationMatrix);
    # display(mutualInformationMatrix)
    # entanglementHeatMap = heatmap(collect(0 : (2 * kMax)), collect(0 : (2 * kMax)), mutualInformationMatrix, 
    #     xticks = xTicks, 
    #     yticks = xTicks, 
    #     lims = (0, 2 * kMax), 
    #     c = cgrad(:Blues), 
    #     aspect_ratio = :equal
    # );
    # display(entanglementHeatMap)

    # # permute rows and columns to match modeOrdering = 0
    # if modeOrdering == 0
    #     indexPermutation = collect(1 : length(momentumModes));
    # elseif modeOrdering == 1
    #     indexPermutation = vcat(collect(2 * kMax : -2 : 2), collect(1 : 2 : (2 * kMax + 1)));
    # end
    # B = mutualInformationMatrix[indexPermutation, indexPermutation];
    # B ./= maximum(B)
    # display(B)

    # # store self information as table [kx ky QMI]
    # entanglementEntropyTable = zeros(Float64, 0, 3);
    # for (idxA, momValA) in enumerate(momentumModes), (idxB, momValB) in enumerate(momentumModes)
    #     if momValA == momValB
    #         entanglementEntropyTable = vcat(entanglementEntropyTable, [idxA idxB B[idxA, idxB]]);
    #     else
    #         entanglementEntropyTable = vcat(entanglementEntropyTable, [idxA idxB 0]);
    #     end
    # end
    
    # fileString = @sprintf("selfInformation_mS_kMax_%d_m_%0.3f_modeOrdering_%d.txt", kMax, m, modeOrdering)
    # columnString = @sprintf("kx\t ky\t QMI\n");
    # open(fileString; write = true) do f
    #     write(f, columnString)
    #     writedlm(f, entanglementEntropyTable)
    # end

    # # store mutual information as table [kx ky QMI]
    # entanglementEntropyTable = zeros(Float64, 0, 3);
    # for (idxA, momValA) in enumerate(momentumModes), (idxB, momValB) in enumerate(momentumModes)
    #     entanglementEntropyTable = vcat(entanglementEntropyTable, [idxA idxB B[idxA, idxB]]);
    # end
    
    # fileString = @sprintf("mutualInformation_mS_kMax_%d_m_%0.3f_modeOrdering_%d.txt", kMax, m, modeOrdering)
    # columnString = @sprintf("kx\t ky\t QMI\n");
    # open(fileString; write = true) do f
    #     write(f, columnString)
    #     writedlm(f, entanglementEntropyTable)
    # end

    # # plot "spectrum" of the mutual information
    # relevantElements = vec(mutualInformationMatrix);
    # # relevantElements = [mutualInformationMatrix[idx, idy] for idx = axes(mutualInformationMatrix, 1), idy = axes(mutualInformationMatrix, 2) if idx != idy];
    # mutualInformationSpectrumPlot = plot(1 : length(relevantElements), sort(relevantElements, rev = true), 
    #     yaxis = :log10, 
    #     xlabel = L"i", 
    #     ylabel = L"\mathcal{I}", 
    #     linewidth = 2.0, 
    #     label = "", 
    #     frame = :box, 
    # )
    # display(mutualInformationSpectrumPlot)

    # # compute local occupation numbers
    # numberOperators = local_number_operators(mS);
    # localOccupations = expectation_values(groundStateMPS, numberOperators);
    # println(localOccupations)

    # # compute ⟨a(-k) a(+k)⟩
    # mpos_AnAn, mpos_CrCr = pairing_operators(mS);
    # expVals_AnAn = zeros(ComplexF64, kMax);
    # expVals_CrCr = zeros(ComplexF64, kMax);
    # for idx = 1 : kMax
    #     expVals_AnAn[idx] = expectation_value_mpo(groundStateMPS, mpos_AnAn[idx]);
    #     expVals_CrCr[idx] = expectation_value_mpo(groundStateMPS, mpos_CrCr[idx]);
    # end
    # println(real.(expVals_AnAn))
    # println(real.(expVals_CrCr))

    # expVals_AnAn = zeros(ComplexF64, kMax);
    # expVals_CrCr = zeros(ComplexF64, kMax);
    # for idx = 1 : kMax
    #     expVals_AnAn[idx] = expectation_value_mpo(excitedStateMPS, mpos_AnAn[idx]);
    #     # expVals_CrCr[idx] = expectation_value_mpo(excitedStateMPS, mpos_CrCr[idx]);
    # end
    # println(real.(expVals_AnAn))
    # # println(real.(expVals_CrCr))

    # # compute local occupation numbers
    # numberOperators = local_number_operators(mS);
    # localOccupations = expectation_values(groundStateMPS, numberOperators);
    # println(localOccupations)

    @printf("energy gap Δ = %0.8f\n", excitedStateEnergy - groundStateEnergy)

end
# display(entanglementEntropyPlot)

# mpsEntanglementEntropies = compute_entanglement_entropies(excitedStateMPS);
# println(mpsEntanglementEntropies)

# localOccupations = expectation_values(excitedStateMPS, numberOperators);
# println(localOccupations)


# # compute ⟨a(-k) a(+k)⟩
# mpos_AnAn, mpos_CrCr = pairing_operators(mS);
# expVals_AnAn = zeros(ComplexF64, kMax);
# expVals_CrCr = zeros(ComplexF64, kMax);
# for idx = 1 : kMax
#     expVals_AnAn[idx] = expectation_value_mpo(groundStateMPS, mpos_AnAn[idx]);
#     expVals_CrCr[idx] = expectation_value_mpo(groundStateMPS, mpos_CrCr[idx]);
# end
# println(real.(expVals_AnAn))
# # println(real.(expVals_CrCr))

# expVals_AnAn = zeros(ComplexF64, kMax);
# expVals_CrCr = zeros(ComplexF64, kMax);
# for idx = 1 : kMax
#     expVals_AnAn[idx] = expectation_value_mpo(excitedStateMPS, mpos_AnAn[idx]);
#     expVals_CrCr[idx] = expectation_value_mpo(excitedStateMPS, mpos_CrCr[idx]);
# end
# println(real.(expVals_AnAn))
# # println(real.(expVals_CrCr))


# # # compute expectation value of vertex operator
# # vertexOperatorExpVal = expectation_value_mpo(groundStateMPS, sG.mpo_H1);

# # get MPS linkDims
# println(getLinkDimsMPS(groundStateMPS))
# # println(getLinkDimsMPS(excitedStateMPS))