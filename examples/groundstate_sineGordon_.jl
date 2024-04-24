#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

# Pkg.activate(".")
using HTTN
using Printf
using TensorKit


# set truncation parameters
truncMethod = 5;
kMax = 1;
nMax = 40;
nMaxZM = 18;
modeOrdering = 1;
bogoliubovR = 1;
bogParameters = 2.5 * ones(Float64, kMax);

# set model parameters
β = 3.4;
λ = 1.0;
L = 25.0;
R = sqrt(4 * π) / β;

# DMRG control parameters
bondDim = 1000;
convTolE = 1e-6;

# use numerical basis optimization
useBasisOptimization = 0;

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM, truncMethod = truncMethod, modeOrdering = modeOrdering, bogoliubovR = bogoliubovR, bogParameters = bogParameters);
hamiltonianParameters = (β = β, R = R, λ = λ, L = L);

# construct Sine-Gordon model with MPO
sG = SineGordonModel(truncationParameters, hamiltonianParameters);
hamMPO = sG.modelMPO;

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);

# initialize MPS
initialMPS = SparseMPS(randn, ComplexF64, physSpaces, virtSpaces);

# run DMRG for ground state
groundStateMPS, groundStateEnergy = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
@printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

# # run DMRG for ground state
# groundStateMPS, groundStateEnergy = find_groundstate(initialMPS, generate_MPO_sG, sG, DMRG2BO(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
# @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

# # run DMRG for excited state
# lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
# excitedStateMPS, excitedStateEnergy = find_excitedstate(groundStateMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
# @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

# # compute entanglement entropies
# mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS);

# # compute local occupation numbers
# numberOperators = local_number_operators(mS);
# localOccupations = expectation_values(groundStateMPS, numberOperators);

# # compute expectation value of vertex operator
# vertexOperatorExpVal = expectation_value_mpo(groundStateMPS, sG.mpo_H1);

# get MPS linkDims
linkDimsMPS_0 = getLinkDimsMPS(groundStateMPS);
# linkDimsMPS_1 = getLinkDimsMPO(excitedState_fM);
display(reshape(linkDimsMPS_0, 1, length(linkDimsMPS_0)))
# display(reshape(linkDimsMPS_1, 1, length(linkDimsMPS_1)))
println("maximal bond dimension of |ψ0(mpo_fullModel)⟩ = $(maximum(linkDimsMPS_0))")
# println("maximal bond dimension of |ψ1(mpo_fullModel)⟩ = $(maximum(linkDimsMPS_1))")
# println("\n")