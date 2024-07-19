#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using HTTN
using LaTeXStrings
using Plots
using Printf
using TensorKit

# set truncation parameters
truncMethod = 5;
kMax = 4;
nMax = 4;
nMaxZM = 24;
modeOrdering = 1;
bogoliubovR = 0;

# set model parameters
θ = 1.0 * π;
m = 0.10;
M = 1 / sqrt(π);
L = 100.0;

# create NamedTuple for truncation and model parameters
truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM, truncMethod = truncMethod, modeOrdering = modeOrdering, bogoliubovR = bogoliubovR);
hamiltonianParameters = (θ = θ, m = m, M = M, L = L);

# construct Schwinger model with MPO
mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters);
physSpaces = mS.physSpaces;

# construct massiveSchwinger MPO
hamMPO = generate_MPO_mS(mS);
println(getLinkDimsMPO(hamMPO))

# initialize vaccum MPS (ground state of non-interacting Hamiltonian)
vacuumMPS = SparseMPS(ones, ComplexF64, physSpaces, fill(U1Space(0 => 1), 2 * kMax + 1 + 1));

# set total time
totalTime = 1.0;

# set number of time steps
δT = 5e-2;
numTimeSteps = Int(totalTime / δT);

# initialize array to store entanglement entropy
storeEntanglementEntropy = zeros(Float64, 0, length(physSpaces) - 1);

# copy input state
timeEvolvedMPS = copy(vacuumMPS);

# perform time evolution with TDVP
for timeStep = 0 : numTimeSteps

    # perform time step
    if timeStep > 0
        timeEvolvedMPS, envL, envR, ϵ = perform_timestep(timeEvolvedMPS, hamMPO, δT, TDVP2(bondDim = 1000, krylovDim = 2, verbosePrint = true));
    end

    # compute entanglement entropies
    mpsEntanglementEntropies = compute_entanglement_entropies(timeEvolvedMPS);
    storeEntanglementEntropy = vcat(storeEntanglementEntropy, reshape(mpsEntanglementEntropies, 1, :));
    # println(mpsEntanglementEntropies)

end

display(storeEntanglementEntropy)

# initialize plot for the entanglement entropy over time
plotEntanglementEntropy = plot(
    xlabel = L"t", 
    zlabel = L"S(T)", 
    frame = :box, 
)

# plot entanglement entropy
momentumModes = getMomentumModes(mS);
for idxB = axes(storeEntanglementEntropy, 2)
    plot!(plotEntanglementEntropy, δT * collect(0 : numTimeSteps), storeEntanglementEntropy[:, idxB],
        linewidth = 2.0, 
        label = "", 
    )
end
display(plotEntanglementEntropy)

# # initialize plot for the entanglement entropy over time
# plotEntanglementEntropy = plot(
#     xlabel = L"t", 
#     ylabel = L"k", 
#     zlabel = L"S(T)", 
#     frame = :box, 
#     camera = (-30, +35), 
# )

# # plot entanglement entropy
# momentumModes = getMomentumModes(mS);
# for idxB = axes(storeEntanglementEntropy, 2)
#     plot!(plotEntanglementEntropy, δT * collect(0 : numTimeSteps), idxB .* ones(numTimeSteps + 1), storeEntanglementEntropy[:, idxB],
#         linewidth = 2.0, 
#         label = "", 
#     )
# end
# display(plotEntanglementEntropy)