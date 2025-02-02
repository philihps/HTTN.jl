using HTTN
using TensorKit
import Distributions: Normal

# set display parameters
modeOrdering = true;

# set modelName
modelName = "massiveSchwinger"

# set model parameters
θ = 1.0 * π;
e = 1.0;
M = e / sqrt(π);
L = 100.0;
fermionMass = 0.1;

# set truncation parameters ### Test kMax = 0
truncMethod = 5;
kMax = 0;
nMax = 5;
nMaxZM = 10;
bogoliubovRot = false;

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (kMax = kMax,
                        nMax = nMax,
                        nMaxZM = nMaxZM,
                        truncMethod = truncMethod,
                        modeOrdering = modeOrdering,
                        bogoliubovRot = bogoliubovRot);
hamiltonianParameters = (θ = θ, m = fermionMass, M = M, L = L)

mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters)
hamMPO = generate_MPO_mS(mS)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = mS.physSpaces;
virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

# Find ground state and ground energy
zeroDim = physSpaces[1].dims.values[1]
hamMat = reshape(convert(Array, hamMPO[1]), (zeroDim, zeroDim))
eigVals, eigVecs = eigen(hamMat)
println("Ground-state energy before transformation: $(eigVals[1])")
groundStateMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1], virtSpaces[2])

# apply Bogoliubov transformation to MPO & MPS
finiteMPS = [groundStateMPS]
ξ = 0.2
println("ξ = $ξ")
hamMPOTransf = updateBogoliubovParameters(mS, [ξ])
hamMPOTransf = generate_MPO_mS(hamMPOTransf)

singleSqOp = singleSqueezingOp(ξ, nMaxZM, physSpaces[1])
@tensor finiteMPS[1][-1 -2; -3] := finiteMPS[1][-1, 1, -3] * singleSqOp[-2, 1]
finiteMPS = normalizeMPS(finiteMPS)

energy = real(expectation_value_mpo(finiteMPS, hamMPOTransf))
println("Energy after transformation: $energy")

# # apply Bogoliubov back transformation to MPO & MPS
# @tensor hamMPOBackTransf[-1 -2; -3 -4] := singleSqOp[2, -4] * 
#                                             hamMPOTransf[1][-1, 1, -3, 2] *
#                                             conj(singleSqOp[1, -2])
# @tensor finiteMPSBackTransf[-1 -2; -3] := conj(singleSqOp[1, -2]) * 
#                                         finiteMPS[1][-1, 1, -3]
# energy = real(expectation_value_mpo([finiteMPSBackTransf], [hamMPOBackTransf]))
# println("Energy after back transformation: $energy")

nothing
