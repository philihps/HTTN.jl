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
modeOrdering = true;

# set truncation parameters
truncMethod = 5;
kMax = 2;
nMax = 3;
nMaxZM = 10;
bogoliubovRot = false;
bogParameters = [1.24, 0.90, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13];
bogParameters = bogParameters[1:kMax];

# set model parameters
β = 0.25 * sqrt(4 * π);
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


groundStateMPS, groundStateEnergy_DMRG = find_groundstate(initialMPS, hamMPO,
DMRG2(; bondDim = 1000,
      truncErr = 1e-6,
      verbosePrint = true))

# metts parameters
numTimeSteps = 500 # 1000
finalBetas = [1000.0, 500.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5];
numMETTS = 100;
# 5.0 didnt converge, 1.0, 0.5 no variance in first 10 samples
finalEnergies = zeros(Float64,length(finalBetas),2)
finalT = [1/i for i in finalBetas]

for (i, finalBeta) in enumerate(finalBetas)
    @show finalBeta
    energies, truncErrs = metts(initialMPS, hamMPO, numTimeSteps, finalBeta,
                            METTS2(; numMETTS = numMETTS, doBasisExtend = false, tol = 1.0)); # energies = -0.1997
    
    _, av_E_last, err_E_last = energies[end, :]
    finalEnergies[i, :] = [av_E_last, err_E_last]
end

aplot = plot();
plot!([finalT[1]; finalT[end]], [groundStateEnergy_DMRG, groundStateEnergy_DMRG], label="Ground state energy")
plot!(finalT, finalEnergies[:, 1], seriestype=:scatter, ls=:dot, label="");
plot!(;
    xlabel=L"T",
    ylabel=L"E_{\mathrm{thermal}}",
    title=L"\beta=0.25"
)
savefig(aplot, OUTPUT_PATH * "low_T_METTS.pdf")

nothing
