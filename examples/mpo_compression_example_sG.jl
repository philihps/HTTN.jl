#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using JLD2
using LinearAlgebra: diagm
using HTTN
using LaTeXStrings
using Plots
using Printf
using TensorKit

# set truncation parameters
modelName = "sineGordon";
truncMethod = 5;
kMax = 6;
nMax = 10;
nMaxZM = 10;
modeOrdering = 1;
bogoliubovR = 1;

# set Bogoliubov rotation parameters
# β = 1.0 * sqrt(4π), L = 15.0
# truncMethod = 5
# nMax = 10, nMaxZM = 10
optimalBogParameters = [[-0.8835909604885277],
                        [-0.9807282051475313, -0.6485006677860818],
                        [-1.0419031449602234, -0.7065637249009884, -0.5217069463899713],
                        [-1.0832070648614078, -0.7451083503206792, -0.5594884149655681,
                         -0.4361570000139321],
                        [-1.112447329438673, -0.7734540934967917, -0.5850305460944181,
                         -0.46218754038059134, -0.3720900965177832],
                        [-1.1335793150876836, -0.7953326598109092, -0.6029189741949903,
                         -0.47977100826593316, -0.3897525013935271, -0.32302633272158393]];
bogParameters = convert.(Float64, optimalBogParameters[kMax]);
# bogParameters = randn(Float64, kMax);

# set model parameters
βFF = sqrt(4 * π);
λ = 1.0;
L = 25.0;

# set list of betas
betaList = βFF * collect(1.0:0.2:1.0);

# flags to compute ground state with DMRG
runDMRG = 0;
verbosity = 0;

# set DMRG parameters
bondDim = 256;
truncErr = 1e-4;
subspaceExpansion = true;

# xaxis settings for the plot of the MPO bond dimension
xLimits = (0, 2 * kMax);
momentumModes = convert.(Int64, collect((-kMax):1:(+kMax)));
if modeOrdering == 1
    momentumModes = sort(momentumModes; by = abs)
end
xTicks = ((collect(0:(2 * kMax))),
          [kVal == 0 ? latexstring(@sprintf("%d", kVal)) :
           latexstring(@sprintf("%+d", kVal))
           for kVal in momentumModes])

# initalize plot for the bond dimension of the MPO
mpoBondDimensionPlot = plot(; xlims = xLimits, xticks = xTicks, xlab = L"k",
                            ylab = L"m_{\textrm{MPO}}(k_L,k_R)", ylims = (0, 260),
                            frame = :box,)

# loop over Hamiltonian parameters
for (idxB, β) in enumerate(betaList)

    # compute compactification radius
    R = sqrt(4 * π) / β

    # create NamedTuple for truncation parameters and model parameters
    truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM,
                            truncMethod = truncMethod, modeOrdering = modeOrdering,
                            bogoliubovR = bogoliubovR, bogParameters = bogParameters)
    hamiltonianParameters = (β = β, R = R, λ = λ, L = L)

    # construct Sine-Gordon model (with MPO)
    sG = SineGordonModel(truncationParameters, hamiltonianParameters)
    display(sG.modeOccupations)

    # construct physical and virtual vector spaces for the MPS
    boundarySpaceL = U1Space(0 => 1)
    boundarySpaceR = U1Space(0 => 1)
    physSpaces = sG.physSpaces
    virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR;
                                     removeDegeneracy = true)

    # construct sineGordon MPO
    hamMPO = generate_MPO_sG(sG)
    mpoBondDims = getLinkDimsMPO(hamMPO)
    println(mpoBondDims)

    plot!(mpoBondDimensionPlot,
          collect(0.5:1:(2 * kMax)),
          mpoBondDims[2:(end - 1)];
          linewidth = 2.0,
          markers = :circle,
          label = "original MPO",)

    if runDMRG == 1

        # initialize vacuum MPS
        # initialMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(sG; modeOrdering = modeOrdering)
        initialMPS = initializeMPS(sG, vacuumMPS; modeOrdering = modeOrdering)

        # run DMRG for ground state
        @time begin
            groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS,
                                                                              hamMPO,
                                                                              DMRG2(;
                                                                                    bondDim = bondDim,
                                                                                    truncErr = truncErr,
                                                                                    subspaceExpansion = subspaceExpansion,
                                                                                    verbosePrint = verbosity,))
        end
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)
    end

    # compress hamMPO
    compressedMPO, Ss = compress_MPO(hamMPO; truncError = 1e-14, verbose = 0)
    mpoBondDims = getLinkDimsMPO(compressedMPO)
    println(mpoBondDims)

    plot!(mpoBondDimensionPlot,
          collect(0.5:1:(2 * kMax)),
          mpoBondDims[2:(end - 1)];
          linewidth = 2.0,
          markers = :circle,
          label = "compressed MPO",)

    # # plot decay of singular values
    # svdPlot = plot(
    #     xlab = "Rank", 
    #     ylab = L"\log_{10}(\lambda)", 
    #     frame = :box, 
    # )
    # for (idxS, S) in enumerate(Ss)
    #     res = Float64[]
    #     for (k, v) in S.data
    #         append!(res, diag(v))
    #     end
    #     sort!(res, rev = true)
    # 	plot!(svdPlot, (1:length(res)), log.(res)/log(10), markers = :circle, label = "Cut $(length(compressedMPO)-idxS)")
    # end
    # display(svdPlot)

    if runDMRG == 1

        # initialize vacuum MPS
        # initialMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(sG; modeOrdering = modeOrdering)
        initialMPS = initializeMPS(sG, vacuumMPS; modeOrdering = modeOrdering)

        # run DMRG for ground state
        @time begin
            groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS,
                                                                              compressedMPO,
                                                                              DMRG2(;
                                                                                    bondDim = bondDim,
                                                                                    truncErr = truncErr,
                                                                                    subspaceExpansion = subspaceExpansion,
                                                                                    verbosePrint = verbosity,))
        end
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)
    end

    display(mpoBondDimensionPlot)
end
