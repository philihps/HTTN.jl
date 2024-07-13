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
truncMethod = 5;
kMax = 3;
nMax = 3;
nMaxZM = 10;
modeOrdering = 1;
bogoliubovR = 0;
bogParameters = [1.24, 1.15, 1.00, 1.02, 0.90, 0.82, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13, 0.12, 0.11, 0.10, 0.10];


# set model parameters
βFF = sqrt(4 * π);
λ = 1.0;
L = 25.0;
# R = sqrt(4 * π) / β;

# set list of betas
betaList = βFF * collect(1.0 : 0.2 : 1.0);

# set DMRG parameters
bondDim = 128;
truncErr = 1e-4;
subspaceExpansion = true;

# use numerical basis optimization
useBasisOptimization = bogoliubovR;

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


# compute ground state with DMRG
runDMRG = 0;

# initalize plot for the bond dimension of the MPO
mpoBondDimensionPlot = plot(
    xlabel = L"i", 
    ylabel = L"m(i)", 
    frame = :box, 
)

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

    # construct physical and virtual vector spaces for the MPS
    boundarySpaceL = U1Space(0 => 1);
    boundarySpaceR = U1Space(0 => 1);
    physSpaces = sG.physSpaces;
    virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);

    # construct sineGordon MPO
    hamMPO = generate_MPO_sG(sG);
    mpoBondDims = getLinkDimsMPO(hamMPO);
    println(mpoBondDims)

    plot!(mpoBondDimensionPlot, mpoBondDims, 
        linewidth = 2.0, 
        markers = :circle, 
        label = "original MPO", 
    )

    if runDMRG == 1

        # initialize vacuum MPS
        # initialMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(sG, vacuumMPS, modeOrdering = modeOrdering);

        # run DMRG for ground state
        @time begin
            groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = true));
        end
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

    end

    # # transform into 3x3 block structure
    # matMPO = convert_MPO_33block(hamMPO);

    # # # convert back to check full MPO
    # # recMPO = convert_33block_MPO(matMPO);

    # # full MPO A
    # boundaryL = TensorMap(ones, one(U1Space()), U1Space(0 => 1));
    # boundaryR = TensorMap(ones, U1Space(0 => 1), one(U1Space()));
    # @tensor mpoTensorA[-1 -2 -3; -4 -5 -6] := boundaryL[1] * hamMPO[1][1, -1, 2, -4] * hamMPO[2][2, -2, 3, -5] * hamMPO[3][3, -3, 4, -6] * boundaryR[4];

    # # full MPO B
    # boundaryL = TensorMap(ones, one(U1Space()), U1Space(0 => 1));
    # boundaryR = TensorMap(ones, U1Space(0 => 1), one(U1Space()));
    # @tensor mpoTensorB[-1 -2 -3; -4 -5 -6] := boundaryL[1] * recMPO[1][1, -1, 2, -4] * recMPO[2][2, -2, 3, -5] * recMPO[3][3, -3, 4, -6] * boundaryR[4];

    # mpoTensorA = convert(Array, mpoTensorA);
    # sizeT = size(mpoTensorA);
    # display(reshape(mpoTensorA, prod(sizeT[1 : 3]), prod(sizeT[4 : 6])))

    # compress hamMPO
    compressedMPO = compress_MPO(hamMPO, truncError = 1e-14);
    mpoBondDims = getLinkDimsMPO(compressedMPO);
    println(mpoBondDims)

    plot!(mpoBondDimensionPlot, mpoBondDims, 
        linewidth = 2.0, 
        markers = :circle, 
        label = "compressed MPO", 
    )


    if runDMRG == 1

        # initialize vacuum MPS
        # initialMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        vacuumMPS = initializeVacuumMPS(sG, modeOrdering = modeOrdering);
        initialMPS = initializeMPS(sG, vacuumMPS, modeOrdering = modeOrdering);

        # run DMRG for ground state
        @time begin
            groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, compressedMPO, DMRG2(bondDim = bondDim, truncErr = truncErr, subspaceExpansion = subspaceExpansion, verbosePrint = true));
        end
        @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

    end

    display(mpoBondDimensionPlot)

end