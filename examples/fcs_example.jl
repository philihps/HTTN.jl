
using Pkg
using Revise

# Pkg.activate(".")
using HTTN
using Plots
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

# create NamedTuple for truncation and model parameters
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

# compute full counting statistics
phiFieldExpVals, frequencies, fourierComponents = compute_phase_distribution(mS,
                                                                             groundStateMPS;
                                                                             verbosePrint = false);
plotFourierComponents = plot(real.(fourierComponents);
                             yaxis = :identity,
                             linewidth = 2,
                             label = "",
                             frame = :box);
display(plotFourierComponents)
