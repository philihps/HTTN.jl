using HTTN
using Printf
using TensorKit
using Test

# set display parameters
modeOrdering = true;

# set modelName
modelName = "massiveSchwinger"

# set truncation parameters
truncMethod = 5;
kMax = 1;
nMax = 5;
nMaxZM = 10;
bogoliubovRot = false;

# set model parameters
θ = 1.0 * π;
e = 1.0;
M = e / sqrt(π);
L = 100.0;
fermionMass = 0.1;

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

# initialize random MPS
initialTensors = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx]
    initialTensors[siteIdx] = randn(ComplexF64, virtSpaces[siteIdx] ⊗ physSpace,
                                    virtSpaces[siteIdx + 1])
end
initialMPS = SparseMPS(initialTensors; normalizeMPS = true);

# set DMRG parameters
bondDim = 128;
truncErr = 1e-6;

# set TDVP parameters
numTimeStep = 500;
finalBeta = 1000.0;
timeRanges = range(0; stop = finalBeta, length = numTimeStep + 1)
timeStep = 1im * (timeRanges[2] - timeRanges[1])
groundStateEnergy_TDVP = 0


@testset "Compare energy" begin
    @info "Start DMRG"
    groundStateMPS, groundStateEnergy_DMRG = find_groundstate(initialMPS, hamMPO,
                                                              DMRG2(; bondDim = 1000,
                                                                    truncErr = 1e-6,
                                                                    verbosePrint = true))
    @show groundStateEnergy_DMRG
    @test isapprox(groundStateEnergy_DMRG, 1.5414652000703606)

    @info "Start TDVP"
    groundStateEnergy_TDVP = 0

    MPS_t = initialMPS
    for tIndex in eachindex(timeRanges)
        MPS_t, envL, envR, ϵ = perform_timestep!(MPS_t, hamMPO, timeStep, TDVP2())
        if tIndex % 100 == 0
            groundStateEnergy_TDVP = expectation_value_mpo(MPS_t, hamMPO)
            println("Energy at $(tIndex)-step: $(groundStateEnergy_TDVP)")
        end
    end

    @test abs(groundStateEnergy_TDVP - groundStateEnergy_DMRG) < 1e-14

    @info "Start METTS"
    _, energies, _ = metts(initialMPS, hamMPO, mS, numTimeStep, finalBeta;
                                 alg = METTS2(; numWarmUp = 3,
                                              numMETTS = 3,
                                              extendBasis = false,
                                              tol = 1.0))

    _, av_E_last, err_E_last = energies[end, :]
    @test abs(groundStateEnergy_TDVP - av_E_last) < 1e-14
end
