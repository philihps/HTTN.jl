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
modelName = "massiveSchwinger"
truncMethod = 5
kMax = 2
nMax = 8
nMaxZM = 12
modeOrdering = false
bogoliubovRot = false

# set model parameters
θ = 1.0 * π
m = 0.10
M = 1 / sqrt(π)
L = 100.0

# initialize Bogoliubov rotation parameters
bogParameters = zeros(ComplexF64, 1 + kMax)

# create NamedTuple for truncation and model parameters
truncationParameters = (kMax = kMax,
                        nMax = nMax,
                        nMaxZM = nMaxZM,
                        truncMethod = truncMethod,
                        modeOrdering = modeOrdering,
                        bogoliubovRot = bogoliubovRot,
                        bogParameters = bogParameters)
hamiltonianParameters = (θ = θ, m = m, M = M, L = L)

# construct Schwinger model with MPO
mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters)
physSpaces = mS.physSpaces
display(physSpaces)

# construct massiveSchwinger MPO
hamMPO = generate_MPO_mS(mS)
println(getLinkDimsMPO(hamMPO))

# construct vertex operator to measure expectation value of ⟨sin(ϕ)⟩
vertexOperator = generate_H1(mS)
println(getLinkDimsMPO(vertexOperator))

# initialize vaccum MPS (ground state of non-interacting Hamiltonian)
vacuumMPS = initializeVacuumMPS(mS; modeOrdering = modeOrdering)

# set total time
totalTime = 5.0

# set number of time steps
δT = 5e-2
numTimeSteps = Int(totalTime / δT)

# set truncation parameters
bondDim = 1024
truncErrT = 1e-6

# initialize array to store entanglement entropy
storeEntanglementEntropy = zeros(Float64, 0, length(physSpaces) - 1)

# initialize array to store bond dimension
storeBondDimension = zeros(Float64, 0, length(physSpaces) - 1)

# initialize array to store energy expectation value
storeEnergy = zeros(Float64, 0, 2)

# initialize array to store vertex operator expectation value
storeVertexOp = zeros(Float64, 0, 2)

# initialize array to store bogParameters
storeBogParameters = zeros(Float64, 1, 1 + 1 + kMax)

# copy input state
timeEvolvedMPS = copy(vacuumMPS)

# perform time evolution with TDVP
for timeStep in 0:numTimeSteps

    @printf("timeStep = %d/%d\n", timeStep, numTimeSteps)

    # perform time step
    if timeStep > 0
        timeEvolvedMPS, envL, envR, ϵ = perform_timestep!(timeEvolvedMPS, hamMPO, δT,
                                                          TDVP2(; bondDim = bondDim,
                                                                truncErrT = truncErrT,
                                                                krylovDim = 2,
                                                                verbosePrint = 1,))

        # perform optimization of squeezing parameters
        if bogoliubovRot
            timeEvolvedMPS, bogParameters, truncationErrors = perform_basisOptimization!(timeEvolvedMPS,
                                                                                         mS,
                                                                                         TDVP2(;
                                                                                               bondDim = bondDim,
                                                                                               truncErrT = truncErrT,
                                                                                               krylovDim = 2,
                                                                                               verbosePrint = 1,))

            # update QFT model
            mS = updateBogoliubovParameters(mS, bogParameters)

            # reconstruct sineGordon MPO
            hamMPO = generate_MPO_mS(mS)

            # reconstruct vertex operator to measure expectation value of ⟨sin(ϕ)⟩
            vertexOperator = generate_H1(mS)

            # store bogParameters
            storeBogParameters = vcat(storeBogParameters, hcat(δT * timeStep, reshape(bogParameters, 1, :)))
        end
    end

    # compute entanglement entropies
    mpsEntanglementEntropies = compute_entanglement_entropies(timeEvolvedMPS)
    storeEntanglementEntropy = vcat(storeEntanglementEntropy,
                                    reshape(mpsEntanglementEntropies, 1, :))
    # println(mpsEntanglementEntropies)

    # get maximal bond dimension
    virtBondDims = getLinkDimsMPS(timeEvolvedMPS)
    storeBondDimension = vcat(storeBondDimension, reshape(virtBondDims[2:(end - 1)], 1, :))
    # println(mpsEntanglementEntropies)

    # # compute energy of time-evolved state
    # energyExpectationValue = expectation_value_mpo(timeEvolvedMPS, hamMPO);
    # storeEnergy = vcat(storeEnergy, [δT * timeStep energyExpectationValue]);

    # compute expectation value of vertex operator
    expValVertexOp = expectation_value_mpo(timeEvolvedMPS, vertexOperator)
    if abs(imag(expValVertexOp)) < 1e-12
        storeVertexOp = vcat(storeVertexOp, [δT * timeStep real(expValVertexOp)])
    else
        ErrorException("vertex operator is not Hermitian, complex expectation value found.")
    end
end

display(storeBondDimension)

# initialize plot for the entanglement entropy over time
plotEntanglementEntropy = plot(; xlabel = L"t", ylabel = L"S(t)", frame = :box)

# plot entanglement entropy
for idxB in axes(storeEntanglementEntropy, 2)
    plot!(plotEntanglementEntropy,
          δT * collect(0:numTimeSteps),
          storeEntanglementEntropy[:, idxB];
          linewidth = 2.0,
          label = "",)
end
display(plotEntanglementEntropy)

# # initialize plot for the entanglement entropy over time
# plotEntanglementEntropy = plot(
#     xlabel = L"t", 
#     ylabel = L"k", 
#     zlabel = L"S(t)", 
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

# initialize plot for the bond dimension over time
plotBondDimensions = plot(; xlabel = L"t", ylabel = L"\chi(t)", frame = :box)

# plot entanglement entropy
for idxB in axes(storeBondDimension, 2)
    plot!(plotBondDimensions,
          δT * collect(0:numTimeSteps),
          storeBondDimension[:, idxB];
          linewidth = 2.0,
          label = "",)
end
display(plotBondDimensions)

# initialize plot for the vertex operator over time
plotVertexOperator = plot(;
                          xlabel = L"t", ylabel = L"\langle \sin(\phi) \rangle",
                          frame = :box)

# plot vertex operator expectation value
plot!(plotVertexOperator,
      storeVertexOp[:, 1],
      storeVertexOp[:, 2];
      linewidth = 2.0,
      label = "",)
display(plotVertexOperator)

if bogoliubovRot

    # initialize plot for the Bogoliubov parameters over time
    plotbogParameters = plot(; xlabel = L"t", ylabel = L"\xi(t)", frame = :box)

    # plot bogParameters
    for idxB in axes(storeBogParameters, 2)
        if idxB > 1
            plot!(plotbogParameters,
                  δT * collect(0:numTimeSteps),
                  storeBogParameters[:, idxB];
                  linewidth = 2.0,
                  label = "",)
        end
    end
    display(plotbogParameters)
end
