#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using JLD
using HTTN
using LaTeXStrings
using Plots
using Printf
using TensorKit

# set modelName
modelName = "sineGordon"

# set truncation parameters
truncMethod = 5;
kMax = 3;
nMax = 3;
nMaxZM = 10;
modeOrdering = 1;
bogoliubovR = 0;
bogParameters = [1.24, 0.90, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13];
bogParameters = bogParameters[1 : kMax];

# set model parameters
β = sqrt(4 * π);
λ = 1.0;
L = 25.0;
R = sqrt(4 * π) / β;

# set DMRG parameters
bondDim = 128;
truncErr = 1e-6;

# # plot entanglement entropy
# xLimits = (0, 2 * kMax);
# momentumModes = convert.(Int64, collect(-kMax : 1 : +kMax));
# if modeOrdering == 1
#     momentumModes = sort(momentumModes, by = abs);
# end
# xTicks = (
#     (collect(0 : (2 * kMax))), [kVal == 0 ? latexstring(@sprintf("%d", kVal)) : latexstring(@sprintf("%+d", kVal)) for kVal in momentumModes]
# )

# # initialize plot for entanglement entropy
# entanglementEntropyPlot = plot( 
#     xlims = xLimits, 
#     xticks = xTicks, 
#     xlab = L"k", 
#     ylims = (0, Inf), 
#     ylab = L"S(k_L,k_R)", 
#     linewidth = 2.0, 
#     legend = :topright, 
#     frame = :box
# );

# mpsEntanglementEntropies = [];
# entanglementEntropyMatrix = [];

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
virtSpaces = fill(U1Space(0 => 1), length(physSpaces) + 1);

# initialize random MPS
initialTensors = Vector{TensorMap}(undef, length(physSpaces));
for siteIdx = eachindex(physSpaces)
    physSpace = physSpaces[siteIdx];
    initialTensors[siteIdx] = TensorMap(randn, U1Space(0 => 1) ⊗ physSpace, U1Space(0 => 1));
end
initialMPS = SparseMPS(initialTensors, normalizeMPS = true);

# construct sineGordon MPO
hamMPO = generate_MPO_sG(sG);

deltaTau = 1e-1;
finalBeta = 0.8;
energies = metts(initialMPS, hamMPO, deltaTau, finalBeta, METTS2(numMETTS = 100, doBasisExtend = false));

plotSamples = plot(energies[:, 1], linewidth = 2.0, frame = :box, xlabel = "METTS sample", ylabel = L"E_{\mathrm{thermal}}", label = "");
plot!(plotSamples, energies[:, 2], color = :black, linewidth = 1.5, label = "");
# plot!(plotSamples, sum(energies)/length(energies) * ones(length(energies)), color = :black, linewidth = 1.5);
display(plotSamples)
