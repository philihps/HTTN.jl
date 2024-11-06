using Pkg

using HTTN
using Printf
using TensorKit
using Test

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
truncationParameters = (
    kMax = kMax,
    nMax = nMax,
    nMaxZM = nMaxZM,
    truncMethod = truncMethod,
    modeOrdering = modeOrdering,
    bogoliubovR = bogoliubovR,
    bogParameters = bogParameters,
);

# set model parameters
β = 0.25 * sqrt(4 * π);
λ = 1.0;
L = 15.0;
R = sqrt(4 * π) / β; # compute compactification radius
hamiltonianParameters = (β = β, R = R, λ = λ, L = L);

# construct Sine-Gordon model with MPO
sG = SineGordonModel(truncationParameters, hamiltonianParameters);
println("Sine-Gordon model: $(sG.modeOccupations)")
hamMPO = generate_MPO_sG(sG);

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(
    sG.physSpaces, boundarySpaceL, boundarySpaceR; removeDegeneracy = true
);

# initialize random MPS
initialTensors = Vector{TensorMap}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx];
    initialTensors[siteIdx] = TensorMap(
        randn, virtSpaces[siteIdx] ⊗ physSpace, virtSpaces[siteIdx + 1]
    );
end

initialMPS = SparseMPS(initialTensors; normalizeMPS = true);

## test DMRG
# set DMRG parameters
bondDim = 128;
truncErr = 1e-6;

groundStateMPS, groundStateEnergy_DMRG = find_groundstate(initialMPS, hamMPO, DMRG2(bondDim = 1000, truncErr = 1e-6, verbosePrint = true));
@printf("ground state energy E0 = %0.8f\n\n", groundStateEnergy)

## test TDVP
numTimeStep = 100;
finalBeta = 1000.0;
timeRanges = range(0, stop=finalBeta / 2, length=numTimeStep+1)
timeStep = 1im*(timeRanges[2] - timeRanges[1])
groundStateEnergy_TDVP = 0

let
    MPS_t = initialMPS
    for tIndex in eachindex(timeRanges)
        MPS_t, envL, envR, ϵ = perform_timestep!(MPS_t, hamMPO, timeStep, TDVP2());
        
        if tIndex % 19 ==0
            global groundStateEnergy_TDVP = expectation_value_mpo(MPS_t, hamMPO);
            println("Energy at $(tIndex)-step: $(groundStateEnergy_TDVP)")
        end
        
    end
end

@test abs(groundStateEnergy_TDVP - groundStateEnergy_DMRG) < 1e-15