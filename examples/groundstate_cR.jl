#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using DelimitedFiles
using HTTN
using JLD2
using KrylovKit
using LaTeXStrings
using Plots
using Printf
using TensorKit

# set truncation parameters
modelName = "coupledRotors"
boundaryConditions = "PBC" # "NBC, "DBC", or "PBC"
M = 8
N = 10

# set model parameters
β = 1.0
ω = sqrt(1.0)
κ = 0.0

# set DMRG parameters
bondDim = 1024
truncErr = 1e-6

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (M = M, N = N)
hamiltonianParameters = (β = β, ω = ω, κ = κ, boundaryConditions = boundaryConditions)

# construct coupled rotors model
cR = CoupledRotorsModel(truncationParameters, hamiltonianParameters)
display(cR.modeOccupations)
display(reshape(cR.physSpaces, 1, :))

# construct physical and virtual vector spaces for the MPS
spaceType = ComplexSpace
boundarySpaceL = oneunit(spaceType)
boundarySpaceR = oneunit(spaceType)
physSpaces = cR.physSpaces
virtSpaces = constructVirtSpaces(cR.physSpaces, boundarySpaceL, boundarySpaceR)

# create coupled rotors Hamiltonian as MPO
hamMPO = generate_MPO_cR(cR)
println(getLinkDimsMPO(hamMPO))

# initialize vacuum ground state
initialMPS = initializeVacuumMPS(cR)
initialMPS = initializeMPS(cR, initialMPS)

# run DMRG for ground state
groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, hamMPO,
                                                                  DMRG2(; bondDim = bondDim,
                                                                        truncErr = truncErr,
                                                                        maxIterationsInit = 10,
                                                                        maxIterations = 2,
                                                                        subspaceExpansion = false,
                                                                        verbosePrint = 1))
@printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

# get MPS bond dimensions
println(getLinkDimsMPS(groundStateMPS))

# compute entanglement entropies
mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS)
println(mpsEntanglementEntropies)

# # plot entanglement entropies
# plotEE = plot(0.5 .+ collect(1:(M - 1)), mpsEntanglementEntropies;
#               xlims = (1, M),
#               xlabel = L"i",
#               ylabel = L"S(p_L \vert p_R)",
#               linewidth = 2,
#               label = "",
#               frame = :box)
