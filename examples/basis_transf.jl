using Random, Distributions
using LinearAlgebra
using TensorKit

using Revise
using HTTN

function normal_in_interval(a::Float64, b::Float64)
    dist = Normal(0, 1)  # Standard normal distribution (mean 0, std 1)
    while true
        x = rand(dist)
        if a <= x <= b
            return x
        end
    end
end

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
display(sG.modeOccupations)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = sG.physSpaces;
virtSpaces = constructVirtSpaces(sG.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

initialTensors = Vector{TensorMap}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx]
    initialTensors[siteIdx] = TensorMap(randn, virtSpaces[siteIdx] ⊗ physSpace,
                                        virtSpaces[siteIdx + 1])
end

initialMPS = SparseMPS(initialTensors; normalizeMPS = true);                                 

hamMPO = generate_MPO_sG(sG);

testMPS = deepcopy(initialMPS)
ξs = Float64[]
sqOps = [] # length 2

for siteIdx in 1 : +1 : (length(testMPS) - 1)
    if mod(siteIdx, 2) == 0
        k = abs(siteIdx ÷ 2)
        nMaxk = sG.modeOccupations[2, :][siteIdx]
        ξ = normal_in_interval(-0.5, 0.5)  # uniform random
        push!(ξs, ξ)
        physSpaceL, physSpaceR = space(testMPS[siteIdx + 0], 2), space(testMPS[siteIdx + 1], 2)
        sqOp = squeezingOp(ξ, nMaxk, -k, k, physSpaceL, physSpaceR)
        push!(sqOps, sqOp)

        @tensor localBond[-1 -2 -3; -4] := sqOp[-2, -3, 1, 3] * testMPS[siteIdx + 0][-1, 1, 2] *
                                           testMPS[siteIdx + 1][2, 3, -4]
        
        U, S, V, ϵ = tsvd(localBond, (1, 2), (3, 4))

        S /= norm(S)
        U = permute(U, (1, 2), (3,))
        V = permute(S * V, (1, 2), (3,))

        testMPS[siteIdx + 0] = U
        testMPS[siteIdx + 1] = V
    end
end
testMPS = normalizeMPS(testMPS)

# eigendecomposition of squeezingOps
modeProjs =  [] # length 2
eigStates = [] # length 2, 4-leg tensor
coeffs = []
for sqOp in sqOps
    D, V = eig(sqOp, (1, 2), (3, 4))
    @tensor proj[-1 -2; -3 -4] := conj(V[-3, -4, 1]) * V[-1, -2, 1]
    @assert proj.colr == proj.rowr
    push!(modeProjs, proj)
    push!(eigStates, V)
    push!(coeffs, D)
end

mpsSample, momSample = sample_MPS_pair!(testMPS, modeProjs)
println("Sample of local basis (index): $(mpsSample)")
println("Sample of local basis (momentum): $(momSample)")

modeSectors = [proj.colr for proj in modeProjs]
blockState = sample_to_BPS(mpsSample, momSample, testMPS, coeffs, modeSectors)

nothing