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

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);
######################################################################

# initialize random MPS
initialTensors = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx]
    initialTensors[siteIdx] = randn(ComplexF64, virtSpaces[siteIdx] ⊗ physSpace,
                                    virtSpaces[siteIdx + 1])
end
initialMPS = SparseMPS(initialTensors; normalizeMPS = true);

@testset "Test auxiliary functions" begin
    mpsSample = [15, 2, 4, 4, 3]
    momSample = [0, -1, 3, -6, 4]

    finiteCPS = sample_to_CPS(mpsSample, momSample, sG)
    spacesString = [string(space(finiteCPS[i], 1)) for i in eachindex(finiteCPS)]

    @test spacesString[2] == "Rep[U₁](0=>1)"
    @test spacesString[3] == "Rep[U₁](-1=>1)"
    @test spacesString[4] == "Rep[U₁](2=>1)"
    @test spacesString[5] == "Rep[U₁](-4=>1)"
end

nothing
