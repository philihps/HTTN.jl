
using Pkg
using Revise

# Pkg.activate(".")
using HTTN
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
m = 0.30;
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

# perform timestep with TDVP
δT = 5e-2;
timeEvolvedMPS, envL, envR, ϵ = perform_timestep(vacuumMPS, hamMPO, δT, TDVP2(bondDim = 1000, krylovDim = 2, verbosePrint = true));