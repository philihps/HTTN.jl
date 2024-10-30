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
modelName = "massiveSchwinger";
truncMethod = 3;
kMax = 4;
nMax = 3;
nMaxZM = 20;
modeOrdering = 1;
bogoliubovR = 0;

# set Bogoliubov rotation parameters
bogParameters = [1.24, 1.15, 1.00, 1.02, 0.90, 0.82, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13, 0.12, 0.11, 0.10, 0.10];

# set model parameters
θ = 1.0 * π;
e = 1.0;
M = e / sqrt(π);
L = 100.0;


# set model parameters
M = 1.
L = 25.0;

# set list of θ
thetaList = π * collect(1.0 : 0.2 : 1.0);

# set list of fermion masses m
fermionMasses = [0.10];

# flags to compute ground state with DMRG
runDMRG = 1;
verbosity = 1;

# set DMRG parameters
bondDim = 256;
truncErr = 1e-4;
subspaceExpansion = true;


# xaxis settings for the plot of the MPO bond dimension
xLimits = (0, 2 * kMax);
momentumModes = convert.(Int64, collect(-kMax : 1 : +kMax));
if modeOrdering == 1
    momentumModes = sort(momentumModes, by = abs);
end
xTicks = (
    (collect(0 : (2 * kMax))), [kVal == 0 ? latexstring(@sprintf("%d", kVal)) : latexstring(@sprintf("%+d", kVal)) for kVal in momentumModes]
)

# initalize plot for the bond dimension of the MPO
mpoBondDimensionPlot = plot(
    xlims = xLimits, 
    xticks = xTicks, 
    xlab = L"k", 
    ylab = L"m_{\textrm{MPO}}(k_L,k_R)", 
    frame = :box, 
)

# loop over Hamiltonian parameters
for (idxT, θ) in enumerate(thetaList), (idxM, m) in enumerate(fermionMasses)

    # create NamedTuple for truncation parameters and model parameters
    truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM, truncMethod = truncMethod, modeOrdering = modeOrdering, bogoliubovR = bogoliubovR, bogParameters = bogParameters);
    hamiltonianParameters = (θ = θ, m = m, M = M, L = L);

    # construct massive Schwinger model (with MPO)
    mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters);
    display(mS.modeOccupations)

    # construct physical and virtual vector spaces for the MPS
    boundarySpaceL = U1Space(0 => 1);
    boundarySpaceR = U1Space(0 => 1);
    physSpaces = mS.physSpaces;
    virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);

    # construct massive Schwinger MPO
    hamMPO = generate_MPO_mS(mS);
    mpoBondDims = getLinkDimsMPO(hamMPO);
    println(mpoBondDims)

    plot!(mpoBondDimensionPlot, collect(0.5 : 1 : (2 * kMax)), mpoBondDims[2 : (end - 1)], 
        linewidth = 2.0, 
        markers = :circle, 
        label = "original MPO", 
    )

    if runDMRG == 1

        # initialize vacuum MPS
        # initialMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(mS, vacuumMPS, modeOrdering = modeOrdering);

        # run DMRG for ground state
        @time begin
            groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = verbosity));
        end
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

    end

    # compress hamMPO
    compressedMPO, Ss = compress_MPO(hamMPO, truncError = 1e-14, verbose = 0);
    mpoBondDims = getLinkDimsMPO(compressedMPO);
    println(mpoBondDims)

    plot!(mpoBondDimensionPlot, collect(0.5 : 1 : (2 * kMax)), mpoBondDims[2 : (end - 1)], 
        linewidth = 2.0, 
        markers = :circle, 
        label = "compressed MPO", 
    )

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
        vacuumMPS = initializeVacuumMPS(mS, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(mS, vacuumMPS, modeOrdering = modeOrdering);

        # run DMRG for ground state
        @time begin
            groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, compressedMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = verbosity));
        end
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

    end

    display(mpoBondDimensionPlot)

end