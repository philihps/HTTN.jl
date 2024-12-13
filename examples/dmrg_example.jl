
using Pkg
using Revise

# Pkg.activate(".")
using HTTN
using Printf
using TensorKit

# set truncation parameters
truncMethod = 5;
kMax = 2;
nMax = 3;
nMaxZM = 24;
modeOrdering = 1;
bogoliubovR = 0;

# set model parameters
θ = 1.0 * π;
m = 0.300;
M = 1 / sqrt(π);
L = 100.0;

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (kMax = kMax, nMax = nMax, nMaxZM = nMaxZM,
                        truncMethod = truncMethod, modeOrdering = modeOrdering,
                        bogoliubovR = bogoliubovR);
hamiltonianParameters = (θ = θ, m = m, M = M, L = L);

# construct Schwinger model with MPO
mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters);
hamMPO = mS.modelMPO;

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = mS.physSpaces;
virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

# initialize MPS
initialMPS = SparseMPS(randn, ComplexF64, physSpaces, virtSpaces);

# run DMRG for ground state
groundStateMPS, groundStateEnergy = find_groundstate(initialMPS, hamMPO,
                                                     DMRG2(; bondDim = 1000,
                                                           truncErr = 1e-6,
                                                           verbosePrint = true));
@printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

# run DMRG for excited state
lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
excitedStateMPS, excitedStateEnergy = find_excitedstate(groundStateMPS, hamMPO,
                                                        lowEnergyStates,
                                                        DMRG2(; bondDim = 1000,
                                                              truncErr = 1e-6,
                                                              verbosePrint = true));
@printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

# compute entanglement entropies
mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS);

# compute local occupation numbers
numberOperators = local_number_operators(mS);
localOccupations = expectation_values(groundStateMPS, numberOperators);

# compute expectation value of vertex operator
vertexOperatorExpVal = expectation_value_mpo(groundStateMPS, mS.mpo_H1);
