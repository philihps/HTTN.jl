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
kMax = 1;
nMax = 5;
nMaxZM = 5;
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
display(mS.modeOccupations)
vacuumMPS = initializeVacuumMPS(mS; modeOrdering = modeOrdering)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = mS.physSpaces;
virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

# initialize random MPS
initialTensors = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx]
    initialTensors[siteIdx] = randn(ComplexF64, virtSpaces[siteIdx] ⊗ physSpace,
                                    virtSpaces[siteIdx + 1])
end
initialMPS = SparseMPS(initialTensors; normalizeMPS = true);

@testset "Test BT 2 modes" begin
    @info "Free part of the Hamiltonian"
    H0MPO = generate_H0(mS)
    boundaryL = TensorMap(ones, one(U1Space()), space(H0MPO[1], 1))
    boundaryR = TensorMap(ones, space(H0MPO[3], 3)', one(U1Space()))

    @tensor H0Mat[-1 -2 -3; -4 -5 -6] := boundaryL[1] * H0MPO[1][1, -1, 2, -4] *
                                         H0MPO[2][2, -2, 3, -5] *
                                         H0MPO[3][3, -3, 4, -6] * boundaryR[4]

    dims = (nMaxZM + 1) * (nMax + 1) * (nMax + 1)
    H0Mat = reshape(convert(Array, H0Mat), dims, dims)
    eigValsFree, eigVecs = eigen(H0Mat)

    ξ = [0.3, 0.1]
    @info "Free part of the transformed Hamiltonian w/ $ξ"
    transfMS = updateBogoliubovParameters(mS; bogoliubovRot = true, bogParameters = ξ)
    H0MPOTransf = generate_H0(transfMS)
    groundStateBTMPS, eigValsFreeTransf = find_groundstate(initialMPS, H0MPOTransf,
                                                           DMRG2(; bondDim = 1000,
                                                                 truncErr = 1e-6,
                                                                 verbosePrint = true))
    println("Ground-state energy after BT transformed H0")
    @test abs(eigValsFreeTransf - eigValsFree[1]) < 1e-8

    @info "Test squeezing operator for GS of free Hamiltonian"
    sqOps = [singleSqueezingOp(-ξ[1], nMaxZM, physSpaces[1]),
             squeezingOp(-ξ[2], nMax, -1, 1, physSpaces[2], physSpaces[3])]
    groundStateSqMPS = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces))
    @tensor groundStateSqMPS[1][-1 -2; -3] := sqOps[1][-2, 1] *
                                              groundStateBTMPS[1][-1, 1, -3]
    @tensor localBond[-1 -2 -3; -4] := sqOps[2][-2, -3, 1, 3] *
                                       groundStateBTMPS[2][-1, 1, 2] *
                                       groundStateBTMPS[3][2, 3, -4]

    U, S, V, _ = tsvd(localBond, ((1, 2), (3, 4)))
    S /= norm(S)
    U = permute(U, ((1, 2), (3,)))
    V = permute(S * V, ((1, 2), (3,)))

    groundStateSqMPS[2] = U
    groundStateSqMPS[3] = V
    groundStateSqMPS = SparseMPS(groundStateSqMPS; normalizeMPS = true)
    println("Overlap of inverse BT GS with vacuum")
    @test abs(real(dotMPS(vacuumMPS, groundStateSqMPS))) - 1.0 < 1e-8

    @info "Full Hamiltonian"
    HMPO = generate_MPO_mS(mS)
    # groundStateMPS, eigValsFull = find_groundstate(initialMPS, HMPO,
    #                                                DMRG2(; bondDim = 1000,
    #                                                      truncErr = 1e-6,
    #                                                      verbosePrint = true))
    boundaryL = TensorMap(ones, one(U1Space()), space(HMPO[1], 1))
    boundaryR = TensorMap(ones, space(HMPO[3], 3)', one(U1Space()))

    @tensor HMat[-1 -2 -3; -4 -5 -6] := boundaryL[1] * HMPO[1][1, -1, 2, -4] *
                                        HMPO[2][2, -2, 3, -5] *
                                        HMPO[3][3, -3, 4, -6] * boundaryR[4]

    dims = (nMaxZM + 1) * (nMax + 1) * (nMax + 1)
    HMat = reshape(convert(Array, HMat), dims, dims)
    eigValsFull, eigVecs = eigen(HMat)
    @show eigValsFull[1:5]

    @info "Full transformed Hamiltonian"
    ξ = [0.1, 0.0]
    transfMS = updateBogoliubovParameters(mS; bogoliubovRot = true, bogParameters = ξ)
    HMPOTransf = generate_MPO_mS(transfMS)
    # groundStateBTMPS, eigValsFullTransf = find_groundstate(initialMPS, HMPOTransf,
    #                                                        DMRG2(; bondDim = 1000,
    #                                                              truncErr = 1e-6,
    #                                                              verbosePrint = true))

    boundaryL = TensorMap(ones, one(U1Space()), space(HMPOTransf[1], 1))
    boundaryR = TensorMap(ones, space(HMPOTransf[3], 3)', one(U1Space()))

    @tensor HMatTransf[-1 -2 -3; -4 -5 -6] := boundaryL[1] * HMPOTransf[1][1, -1, 2, -4] *
                                              HMPOTransf[2][2, -2, 3, -5] *
                                              HMPOTransf[3][3, -3, 4, -6] * boundaryR[4]

    dims = (nMaxZM + 1) * (nMax + 1) * (nMax + 1)
    HMatTransf = reshape(convert(Array, HMatTransf), dims, dims)
    eigValsFullTransf, eigVecs = eigen(HMatTransf)
    @show eigValsFullTransf[1:5]

    # @show eigValsFull
    # @show eigValsFullTransf
    # println("Ground-state energy after BT transformed H")
    # @test abs(eigValsFullTransf - eigValsFull) < 1e-8

    # @info "Test squeezing operator for GS of full Hamiltonian"
    # groundStateInvTransf = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces))
    # @tensor groundStateInvTransf[1][-1 -2; -3] := sqOps[1][-2, 1] *
    #                                               groundStateBTMPS[1][-1, 1, -3]
    # @tensor localBond[-1 -2 -3; -4] := sqOps[2][-2, -3, 1, 3] *
    #                                    groundStateBTMPS[2][-1, 1, 2] *
    #                                    groundStateBTMPS[3][2, 3, -4]

    # U, S, V, _ = tsvd(localBond, ((1, 2), (3, 4)))
    # S /= norm(S)
    # U = permute(U, ((1, 2), (3,)))
    # V = permute(S * V, ((1, 2), (3,)))

    # groundStateInvTransf[2] = U
    # groundStateInvTransf[3] = V
    # groundStateInvTransf = SparseMPS(groundStateInvTransf; normalizeMPS = true)

    # println("Overlap of inverse BT GS with original GS")
    # @test abs(real(dotMPS(groundStateMPS, groundStateInvTransf))) - 1.0 < 1e-8
end

nothing
