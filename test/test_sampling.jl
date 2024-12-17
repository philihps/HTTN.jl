using HTTN
using TensorKit
using Test

# set modelName
modelName = "sineGordon"

# set display parameters
modeOrdering = true;

# set truncation parameters
truncMethod = 5;
kMax = 2;
nMax = 5;
nMaxZM = 10;
bogoliubovRot = false;
bogParameters = [1.24, 0.90, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13];
bogParameters = bogParameters[1:kMax];

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
                        bogoliubovRot = bogoliubovRot,
                        bogParameters = bogParameters);
hamiltonianParameters = (θ = θ, m = fermionMass, M = M, L = L)

mS = MassiveSchwingerModel(truncationParameters, hamiltonianParameters)
hamMPO = generate_MPO_mS(mS)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = mS.physSpaces;
virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

######################################################################
nTrials = 100

@testset "Test total momentum preservation of sampling" begin
    for trial in (1:nTrials)

        # initialize random MPS
        testMPS = Vector{TensorMap}(undef, length(physSpaces))
        for siteIdx in eachindex(physSpaces)
            physSpace = physSpaces[siteIdx]
            testMPS[siteIdx] = TensorMap(randn, virtSpaces[siteIdx] ⊗ physSpace,
                                         virtSpaces[siteIdx + 1])
        end

        testMPS = SparseMPS(testMPS; normalizeMPS = true)
        testMPS, sqOps = transform_basis!(testMPS, mS)
        testMPS_1 = deepcopy(testMPS)

        mpsSample, momSample = sample_MPS!(testMPS)
        @test sum(momSample) == 0

        mpsSample_1, momSample_1 = sample_MPS_block!(testMPS_1, sqOps)
        @test sum(momSample_1) == 0
    end
end
