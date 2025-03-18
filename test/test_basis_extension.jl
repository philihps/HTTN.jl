using HTTN
using TensorKit
using Test

# set modelName
modelName = "massiveSchwinger"

# set display parameters
modeOrdering = true;

# set truncation parameters
truncMethod = 3;
kMax = 2;
nMax = 3;
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

# construct Sine-Gordon model (with MPO)
mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters)
display(mS.modeOccupations)
hamMPO = generate_MPO_mS(mS)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);
fullVirtSpaces = [virtSpace.dims.keys for virtSpace in virtSpaces]

@testset "Test basis extension" begin

    # initialize random product state
    mpsSample = [5, 2, 4, 4, 3]
    momSample = [0, -1, 3, -6, 4]

    finiteMPS = sample_to_CPS(mpsSample, momSample, mS)
    virtSpacesCPS = vcat([space(finiteMPS[1], 1).dims.keys],
                         [space(finiteMPS[i], 3).dims.keys for i in 1:length(physSpaces)])
    @show virtSpacesCPS

    timeStep = 0.05
    finiteMPS, _, _, _ = perform_timestep!(finiteMPS, hamMPO, timeStep, TDVP2())

    virtSpacesMPS = vcat([space(finiteMPS[1], 1).dims.keys],
                         [space(finiteMPS[i], 3).dims.keys for i in 1:length(physSpaces)])
    @test any(length(virtSpacesMPS[i]) > length(virtSpacesCPS[i])
              for i in eachindex(virtSpacesCPS))
end

nothing
