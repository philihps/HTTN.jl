using HTTN
using TensorKit
using Test

# set modelName
modelName = "sineGordon"

# set display parameters
modeOrdering = true;

# set truncation parameters
truncMethod = 3;
kMax = 2;
nMax = 3;
nMaxZM = 10;
bogoliubovRot = false;

# set model parameters
β = 0.25 * sqrt(4 * π);
λ = 1.0;
L = 15.0;

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (kMax = kMax,
                        nMax = nMax,
                        nMaxZM = nMaxZM,
                        truncMethod = truncMethod,
                        modeOrdering = modeOrdering,
                        bogoliubovRot = bogoliubovRot);
hamiltonianParameters = (β = β, λ = λ, L = L);

# construct Sine-Gordon model (with MPO)
sG = SineGordonModel(truncationParameters, hamiltonianParameters);
display(sG.modeOccupations)
hamMPO = generate_MPO_sG(sG)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);
fullVirtSpaces = [virtSpace.dims.keys for virtSpace in virtSpaces]

@testset "Test basis extension" begin

    # initialize random product state
    mpsSample = [15, 2, 4, 4, 3]
    momSample = [0, -1, 3, -6, 4]

    finiteMPS = sample_to_CPS(mpsSample, momSample, sG)
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
