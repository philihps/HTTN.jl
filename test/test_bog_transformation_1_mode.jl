using HTTN
using TensorKit

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
nMax = 3;
nMaxZM = 100;
@show nMaxZM
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

vacuumMPS = initializeVacuumMPS(mS; modeOrdering = modeOrdering)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = mS.physSpaces;
virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

## test free part
# GS energy of untransformed H_free
println("Info: Free like the Free University")
zeroDim = physSpaces[1].dims.values[1]
H0MPO = generate_H0(mS)
H0Mat = reshape(convert(Array, H0MPO[1]), (zeroDim, zeroDim))
eigVals, eigVecs = eigen(H0Mat)
groundStateMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1], virtSpaces[2])
groundStateMPS = SparseMPS([groundStateMPS]; normalizeMPS = true)
println("Ground-state energy before transformation: $(eigVals[1])")
println("First excited state energy before transformation: $(eigVals[2])")
println("Vacuum overlap before transformation:: $(dotMPS(groundStateMPS, vacuumMPS))")

# GS energy of transformed H_free
ξ = 0.3
println("#########################")
@show ξ
transfMS = updateBogoliubovParameters(mS; bogoliubovRot = true, bogParameters = [ξ])
H0MPOTransf = generate_H0(transfMS)
H0MatTransf = reshape(convert(Array, H0MPOTransf[1]), (zeroDim, zeroDim))
eigVals, eigVecs = eigen(H0MatTransf)
groundStateBTMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1], virtSpaces[2])
groundStateBTMPS = SparseMPS([groundStateBTMPS]; normalizeMPS = true)
println("Ground-state energy after BT transformation of H0: $(eigVals[1])")
println("First excited state energy after BT transformation of H0: $(eigVals[2])")
println("Vacuum overlap after transformation: $(dotMPS(groundStateBTMPS, vacuumMPS))")

# Apply squeezing operator to groundStateMPS
singleSqOp = singleSqueezingOp(ξ, nMaxZM, physSpaces[1])
@tensor groundStateSQMPS[-1 -2; -3] := singleSqOp[-2, 1] *
                                       groundStateMPS[1][-1, 1, -3]
normSQState = real(tr(groundStateSQMPS' * groundStateSQMPS))
groundStateSQMPS = SparseMPS([groundStateSQMPS]; normalizeMPS = false)
println("Norm of the squeezed (vacuum) state: $normSQState")
groundStateSQMPS[1] /= norm(groundStateSQMPS[1])
println("Overlap between squeezed and BT ground state: $(dotMPS(groundStateSQMPS, groundStateBTMPS))")

## test interaction part
println("#########################")
println("Info: Oh no interaction")
H1MPO = generate_H1(mS)
H1Mat = reshape(convert(Array, H1MPO[1]), (zeroDim, zeroDim))
eigVals, eigVecs = eigen(H1Mat)
groundStateMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1], virtSpaces[2])
groundStateMPS = SparseMPS([groundStateMPS]; normalizeMPS = true)
println("Ground-state energy before transformation: $(eigVals[1])")
println("First excited state energy before transformation: $(eigVals[2])")
println("#########################")

ξ = 0.1
@show ξ
transfMS = updateBogoliubovParameters(mS; bogoliubovRot = true, bogParameters = [ξ])
H1MPOTransf = generate_H1(transfMS)
H1MatTransf = reshape(convert(Array, H1MPOTransf[1]), (zeroDim, zeroDim))
eigVals, eigVecs = eigen(H1MatTransf)
groundStateBTMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1], virtSpaces[2])
groundStateBTMPS = SparseMPS([groundStateBTMPS]; normalizeMPS = true)
println("Ground-state energy after BT transformation of H1: $(eigVals[1])")
println("First excited state energy after BT transformation of H1: $(eigVals[2])")

singleSqOp = singleSqueezingOp(ξ, nMaxZM, physSpaces[1])
@tensor groundStateInvTransf[-1 -2; -3] := conj(singleSqOp[1, -2]) *
                                           groundStateBTMPS[1][-1, 1, -3]
groundStateInvTransf /= norm(groundStateInvTransf)
energy = real(expectation_value_mpo([groundStateInvTransf], H1MPO))
println("Ground-state energy after inverse transformation of BT transformed GS + normalization: $(energy)")

ξ = -0.1
@show ξ
transfMS = updateBogoliubovParameters(mS; bogoliubovRot = true, bogParameters = [ξ])
H1MPOTransf = generate_H1(transfMS)
H1MatTransf = reshape(convert(Array, H1MPOTransf[1]), (zeroDim, zeroDim))
eigVals, eigVecs = eigen(H1MatTransf)
println("Ground-state energy after BT transformation of H1: $(eigVals[1])")
println("First excited state energy after BT transformation of H1: $(eigVals[2])")
