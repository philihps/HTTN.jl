using HTTN
using TensorKit
using Test

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
nMax = 10;
nMaxZM = 20;
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
zeroDim = physSpaces[1].dims.values[1]
H0MPO = generate_H0(mS)
H0Mat = reshape(convert(Array, H0MPO[1]), (zeroDim, zeroDim))

@testset "Test BT 1 mode" begin
    @info "Free part of the Hamiltonian"
    eigValsFree, eigVecs = eigen(H0Mat)
    groundStateMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1], virtSpaces[2])
    groundStateMPS = SparseMPS([groundStateMPS]; normalizeMPS = true)

    ξ = 0.3
    @info "Free part of the transformed Hamiltonian w/ $ξ"
    transfMS = updateBogoliubovParameters(mS; bogoliubovRot = true, bogParameters = [ξ])
    H0MPOTransf = generate_H0(transfMS)
    H0MatTransf = reshape(convert(Array, H0MPOTransf[1]), (zeroDim, zeroDim))
    eigValsFreeTransf, eigVecs = eigen(H0MatTransf)
    groundStateBTMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1],
                                 virtSpaces[2])
    @test abs(norm(groundStateBTMPS) - 1.0) < 1e-8
    print("Overlap of GS with vacuum")
    @test abs(real(dotMPS(groundStateMPS, vacuumMPS)) - 1.0) < 1e-8

    groundStateBTMPS = SparseMPS([groundStateBTMPS]; normalizeMPS = true)
    println("Ground-state energy after BT transformed H0")
    @test abs(eigValsFreeTransf[1] - eigValsFree[1]) < 1e-8
    println("First excited state energy after BT transformed H0")
    @test abs(eigValsFreeTransf[2] - eigValsFree[2]) < 1e-8

    @info "Test squeezing operator for GS of free Hamiltonian"
    singleSqOp = singleSqueezingOp(-ξ, nMaxZM, physSpaces[1])
    @tensor groundStateSqMPS[-1 -2; -3] := singleSqOp[-2, 1] *
                                           groundStateBTMPS[1][-1, 1, -3]
    normSQState = real(tr(groundStateSqMPS' * groundStateSqMPS))
    println("Norm of the squeezed (vacuum) state: $normSQState")
    groundStateSqMPS = SparseMPS([groundStateSqMPS]; normalizeMPS = true)
    println("Overlap of inverse BT GS with vacuum")
    @test abs(real(dotMPS(vacuumMPS, groundStateSqMPS))) - 1.0 < 1e-8

    @info "Full Hamiltonian"
    HMPO = generate_MPO_mS(mS)
    HMat = reshape(convert(Array, HMPO[1]), (zeroDim, zeroDim))
    eigValsFull, eigVecs = eigen(HMat)
    groundStateMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1], virtSpaces[2])
    groundStateMPS = SparseMPS([groundStateMPS]; normalizeMPS = true)

    @info "Full transformed Hamiltonian"
    ξ = 0.3
    transfMS = updateBogoliubovParameters(mS; bogoliubovRot = true, bogParameters = [ξ])
    HMPOTransf = generate_MPO_mS(transfMS)
    HMatTransf = reshape(convert(Array, HMPOTransf[1]), (zeroDim, zeroDim))
    eigValsFullBT, eigVecs = eigen(HMatTransf)
    groundStateBTMPS = TensorMap(eigVecs[:, 1], virtSpaces[1] ⊗ physSpaces[1],
                                 virtSpaces[2])
    groundStateBTMPS = SparseMPS([groundStateBTMPS]; normalizeMPS = true)
    println("Ground-state energy after BT transformed H")
    @test abs(eigValsFullBT[1] - eigValsFull[1]) < 1e-8
    println("First excited state energy after BT transformed H")
    @test abs(eigValsFullBT[2] - eigValsFull[2]) < 1e-8

    @info "Test squeezing operator for GS of full Hamiltonian"
    singleSqOp = singleSqueezingOp(-ξ, nMaxZM, physSpaces[1])
    @tensor groundStateInvTransf[-1 -2; -3] := singleSqOp[-2, 1] *
                                               groundStateBTMPS[1][-1, 1, -3]
    groundStateInvTransf = SparseMPS([groundStateInvTransf]; normalizeMPS = true)
    energy = real(expectation_value_mpo(groundStateInvTransf, HMPO))

    println("Energy of inverse transformed GS")
    @test abs(energy - eigValsFull[1]) < 1e-8
    println("Overlap between original GS and inverse transformed GS:")
    @test abs(dotMPS(groundStateMPS, groundStateInvTransf) - 1.0) < 1e-8
end

nothing
