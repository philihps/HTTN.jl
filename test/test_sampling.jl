using HTTN
using TensorKit
using Test
using Revise

# set modelName
modelName = "sineGordon"

# set display parameters
modeOrdering = true;

# set truncation parameters
truncMethod = 5;
kMax = 2;
nMax = 3;
nMaxZM = 10;
bogoliubovRot = false;
bogParameters = [1.24, 0.90, 0.71, 0.60, 0.55, 0.45, 0.39, 0.29, 0.25, 0.21, 0.17, 0.13];
bogParameters = bogParameters[1:kMax];

# set model parameters
β = 0.25 * sqrt(4 * π); # frequency unit
invβ = 1 / β;
R = 1/β
λ = 1.0;
L = 15.0;

# create NamedTuple for truncation parameters and model parameters
truncationParameters = (kMax = kMax,
                        nMax = nMax,
                        nMaxZM = nMaxZM,
                        truncMethod = truncMethod,
                        modeOrdering = modeOrdering,
                        bogoliubovRot = bogoliubovRot,
                        bogParameters = bogParameters);
hamiltonianParameters = (β = β, λ = λ, L = L);

# construct Sine-Gordon model (with MPO)
sG = SineGordonModel(truncationParameters, hamiltonianParameters);
hamMPO = generate_MPO_sG(sG);

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

######################################################################
nTrials = 100

@testset "Test total momentum preservation of sampling" begin
    for trial in (1 : nTrials)
            
        # initialize random MPS
        testMPS = Vector{TensorMap}(undef, length(physSpaces));
        for siteIdx in eachindex(physSpaces)
            physSpace = physSpaces[siteIdx]
            testMPS[siteIdx] = TensorMap(randn, virtSpaces[siteIdx] ⊗ physSpace,
                                                virtSpaces[siteIdx + 1])
        end

        testMPS = SparseMPS(testMPS; normalizeMPS = true);
        testMPS, eigStates, eigVals = transform_basis!(testMPS, sG)
        
        mpsSample, momSample = sample_MPS!(testMPS)
        @test sum(momSample) == 0

        if trial == 1
            # Approach 1: initialize state as eigenstate of squeezing operator
            modeSectors = [eigState.colr for eigState in eigStates]
            interState = sample_to_BPS(mpsSample, momSample, sG, eigVals, modeSectors)
            blockState = Vector{TensorMap}(undef, length(physSpaces));
            blockState[1] = interState[1]
            for i in 1:(length(interState)-1)
                U, S, V, _ = tsvd(interState[i+1], (1, 2), (3, 4))
                S /= norm(S)
                U = permute(U * sqrt(S), (1, 2), (3,))
                V = permute(sqrt(S) * V, (1, 2), (3,))

                blockState[2*i + 0] = U
                blockState[2*i + 1] = V
            end
            blockState = SparseMPS(blockState, normalizeMPS = true)
            blockStateEnergy = expectation_value_mpo(blockState, hamMPO)

            # Approach 2: initialize state as CPS in new basis and apply inverse transformation
            productState = sample_to_CPS(mpsSample, momSample, sG)

            for siteIdx in eachindex(testMPS)
                if mod(siteIdx, 2) == 0
                    eigState = eigStates[siteIdx ÷ 2]

                    @tensor localBond[-1 -2 -3; -4] := eigState'[-2, -3, 1, 3] * productState[siteIdx + 0][-1, 1, 2] *
                                                        productState[siteIdx + 1][2, 3, -4]
                    
                    U, S, V, ϵ = tsvd(localBond, (1, 2), (3, 4))

                    S /= norm(S)
                    U = permute(U, (1, 2), (3,))
                    V = permute(S * V, (1, 2), (3,))

                    productState[siteIdx + 0] = U
                    productState[siteIdx + 1] = V
                end
            end

            productState = normalizeMPS(productState)
            productStateEnergy = expectation_value_mpo(productState, hamMPO)
            @test abs(blockStateEnergy - productStateEnergy) < 1e-10
        end
    end
end

