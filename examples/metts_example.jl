#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

Pkg.activate(".")
using JLD
using HTTN
using LaTeXStrings
using Plots
using Printf
using TensorKit

# plot settings
OUTPUT_PATH = "/home/psireal42/study/HTTN.jl/outputs/"
default(; fontfamily = "Computer Modern")
colorPal = palette(:tab10)

# set modelName
modelName = "sineGordon"

# set display parameters
modeOrdering = 1;

# set truncation parameters
truncMethod = 5;
kMax = 4;
nMax = 4;
nMaxZM = 10;
bogoliubovRot = false;
bogParameters = [1.24, 0.90, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13];
bogParameters = bogParameters[1:kMax];

# set model parameters
β = 1.0 * sqrt(4 * π);
λ = 1.0;
L = 15.0;

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (kMax = kMax,
                        nMax = nMax,
                        nMaxZM = nMaxZM,
                        truncMethod = truncMethod,
                        modeOrdering = modeOrdering,
                        bogoliubovRot = bogoliubovRot,
                        bogParameters = bogParameters);
hamiltonianParameters = (β = β, λ = λ, L = L);

# construct Sine-Gordon model (with MPO)
sG = SineGordonModel(truncationParameters, hamiltonianParameters);
display(sG.modeOccupations)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

# initialize random MPS
initialTensors = Vector{TensorMap}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx]
    initialTensors[siteIdx] = TensorMap(randn, virtSpaces[siteIdx] ⊗ physSpace,
                                        virtSpaces[siteIdx + 1])
end

initialMPS = SparseMPS(initialTensors; normalizeMPS = true);

# construct sineGordon MPO
hamMPO = generate_MPO_sG(sG);
numTimeStep = 100;
finalBeta = 20.0;
numMETTS = 5;
energies, truncErrs = metts(initialMPS, hamMPO, numTimeStep, finalBeta,
                            METTS2(; numMETTS = numMETTS, doBasisExtend = false, tol = 5.0)); # energies = -0.1997

# aplot = plot();
# plot!((1:numMETTS),energies[:, 1], label="concurrent");
# plot!((1:numMETTS),energies[:, 2], yerror=energies[:,3], label="average");
# plot!(;
#     xlabel="No. of METTS samples",
#     ylabel=L"E_{\mathrm{thermal}}",
#     legend=:topleft,
#     title=L"\beta=%$(finalBeta), \tau=%$timeStep"
# )
# savefig(aplot, OUTPUT_PATH * "low_T_METTS.pdf")

nothing
numTimeStep = 10
finalBeta = 2.0;
energies = metts(initialMPS,
                 hamMPO,
                 numTimeStep,
                 finalBeta,
                 METTS2(; numMETTS = numTimeStep, doBasisExtend = true));

plotSamples = plot(energies[:, 1];
                   linewidth = 2.0,
                   frame = :box,
                   xlabel = "METTS sample",
                   ylabel = L"E_{\mathrm{thermal}}",
                   label = "",);
plot!(plotSamples, energies[:, 2]; color = :black, linewidth = 1.5, label = "");
# plot!(plotSamples, sum(energies)/length(energies) * ones(length(energies)), color = :black, linewidth = 1.5);
display(plotSamples)
# println("ok")

# sampleResult, sampleMomentum = sample_from_MPS!(initialMPS);
# display(reshape(sampleResult, 1, :))
# display(reshape(sampleMomentum, 1, :))
# println("sum of sampled momenta = ", sum(sampleMomentum))
