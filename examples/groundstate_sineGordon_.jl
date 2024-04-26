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
kMax = 4;
nMax = 4;
nMaxZM = 18;
modeOrdering = 1;
bogoliubovR = 1;
bogParameters = 0.4 .+ 0.1 * reverse(collect(1 : kMax));
# bogParameters = [1.25, 0.7658787726053576, 0.5916149758233501, 0.46691233917615466, 0.37809892346039614, 0.31132850050333677];
bogParameters = [ 1.0730311449195271, 0.7349048939229015, 0.5492985081346116, 0.42663356984035494];

# set model parameters
β = 3.5;
λ = 1.0;
L = 25.0;
R = sqrt(4 * π) / β;

# DMRG control parameters
bondDim = 1000;
convTolE = 1e-6;

# use numerical basis optimization
useBasisOptimization = bogoliubovR;

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
virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR, removeDegeneracy = true);

# initialize MPS
initialMPS = SparseMPS(randn, ComplexF64, physSpaces, virtSpaces);

if useBasisOptimization == 0

    # construct sineGordon MPO
    hamMPO = generate_MPO_sG(sG);
    println(getLinkDimsMPO(hamMPO))

    # run DMRG for ground state
    groundStateMPS, groundStateEnergy = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
    @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

elseif useBasisOptimization == 1

    # run DMRG for ground state
    groundStateMPS, groundStateEnergy = find_groundstate(initialMPS, generate_MPO_sG, sG, DMRG2BO(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
    @printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

end

# # # run DMRG for excited state
# # lowEnergyStates = Vector{SparseMPS}([groundStateMPS]);
# # excitedStateMPS, excitedStateEnergy = find_excitedstate(groundStateMPS, hamMPO, lowEnergyStates, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
# # @printf("excited state energy E1 = %0.8f\n\n", excitedStateEnergy)

# compute entanglement entropies
mpsEntanglementEntropies = compute_entanglement_entropies(groundStateMPS);
println(mpsEntanglementEntropies)

# compute local occupation numbers
numberOperators = local_number_operators(sG);
localOccupations = expectation_values(groundStateMPS, numberOperators);
println(localOccupations)


# compute ⟨a(-k) a(+k)⟩
mpos_AnAn, mpos_CrCr = pairing_operators(sG);
expVals_AnAn = zeros(ComplexF64, kMax);
expVals_CrCr = zeros(ComplexF64, kMax);
for idx = 1 : kMax
    expVals_AnAn[idx] = expectation_value_mpo(groundStateMPS, mpos_AnAn[idx]);
    expVals_CrCr[idx] = expectation_value_mpo(groundStateMPS, mpos_CrCr[idx]);
end
println(real.(expVals_AnAn))
println(real.(expVals_CrCr))


# # # compute expectation value of vertex operator
# # vertexOperatorExpVal = expectation_value_mpo(groundStateMPS, sG.mpo_H1);

# get MPS linkDims
println(getLinkDimsMPS(groundStateMPS))
# linkDimsMPS_0 = getLinkDimsMPS(groundStateMPS);
# linkDimsMPS_1 = getLinkDimsMPO(excitedState_fM);
# display(reshape(linkDimsMPS_0, 1, length(linkDimsMPS_0)))
# display(reshape(linkDimsMPS_1, 1, length(linkDimsMPS_1)))
# println("maximal bond dimension of |ψ0(mpo_fullModel)⟩ = $(maximum(linkDimsMPS_0))")
# println("maximal bond dimension of |ψ1(mpo_fullModel)⟩ = $(maximum(linkDimsMPS_1))")
# println("\n")