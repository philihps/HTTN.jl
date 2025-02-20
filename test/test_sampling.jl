using HTTN
using TensorKit
using Test

# set modelName
modelName = "massiveSchwinger"

# set display parameters
modeOrdering = true;

# set truncation parameters
truncMethod = 5;
kMax = 2;
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

######################################################################
nTrials = 50

@testset "Test total momentum preservation of sampling" begin
    for trial in (1:nTrials)

        # initialize random MPS
        testMPS = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces))
        for siteIdx in eachindex(physSpaces)
            physSpace = physSpaces[siteIdx]
            testMPS[siteIdx] = randn(ComplexF64, virtSpaces[siteIdx] ⊗ physSpace,
                                     virtSpaces[siteIdx + 1])
        end

        testMPS = SparseMPS(testMPS; normalizeMPS = true)
        testMPS, sqOps = transform_basis!(testMPS, mS; squeezeZM = true,
                                          squeezeNonZM = true, transfWidth = 0.1)
        testMPS_1 = deepcopy(testMPS)

        mpsSample, momSample = sample_MPS!(testMPS)
        @test sum(momSample) == 0

        mpsSample_1, momSample_1 = sample_MPS_block!(testMPS_1, sqOps)
        @test sum(momSample_1) == 0
    end
end
