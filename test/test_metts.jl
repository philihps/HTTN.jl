using HTTN
using Printf
using TensorKit
using Test

# set modelName
modelName = "sineGordon"

# set display parameters
modeOrdering = 1;

# set truncation parameters
truncMethod = 3;
kMax = 2;
nMax = 3;
nMaxZM = 10;
bogoliubovR = 0;
bogParameters = [1.24, 0.90, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13];
bogParameters = bogParameters[1:kMax];

# set model parameters
β = 0.25 * sqrt(4 * π);
λ = 1.0;
L = 15.0;
R = sqrt(4 * π) / β;

# set DMRG parameters
bondDim = 128;
truncErr = 1e-6;

# compute compactification radius
R = sqrt(4 * π) / β;

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (
    kMax = kMax,
    nMax = nMax,
    nMaxZM = nMaxZM,
    truncMethod = truncMethod,
    modeOrdering = modeOrdering,
    bogoliubovR = bogoliubovR,
    bogParameters = bogParameters,
);
hamiltonianParameters = (β = β, R = R, λ = λ, L = L);

# construct Sine-Gordon model (with MPO)
sG = SineGordonModel(truncationParameters, hamiltonianParameters);
display(sG.modeOccupations)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(
    sG.physSpaces, boundarySpaceL, boundarySpaceR; removeDegeneracy = false
);
######################################################################

# initialize random MPS
initialTensors = Vector{TensorMap}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx];
    initialTensors[siteIdx] = TensorMap(
        randn, virtSpaces[siteIdx] ⊗ physSpace, virtSpaces[siteIdx + 1]
    );
end
initialMPS = SparseMPS(initialTensors; normalizeMPS = true);


# test sample_to_CPS

@testset "Helper functions" begin
    mpsSample = [15, 2, 4, 4, 3]
    momSample = [0, -1, 3, -6, 4]

    finiteCPS = sample_to_CPS(mpsSample, momSample, initialMPS)
    spacesString = [string(space(finiteCPS[i], 3)) for i in eachindex(finiteCPS)]
    @test spacesString[2] == "Rep[U₁](0=>1)"
    @test spacesString[3] == "Rep[U₁](-1=>1)"
    @test spacesString[4] == "Rep[U₁](2=>1)"
    @test spacesString[5] == "Rep[U₁](-4=>1)"
    
end


nothing