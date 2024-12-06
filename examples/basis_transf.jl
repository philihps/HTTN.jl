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

# create eigenmodes
mpsSample1 = [15, 4, 2, 1, 2]
momSample1 = [0, -3, 1, 0, 2]
mpsSample2 = [10, 3, 3, 2, 2]
momSample2 = [0, -2, 2, -2, 2]

eigenMode1 = sample_to_CPS(mpsSample1, momSample1, initialMPS)
eigenMode2 = sample_to_CPS(mpsSample2, momSample2, initialMPS)
eigenMode = eigenMode1 + eigenMode2
eigenMode = normalizeMPS(eigenMode)


hamMPO = generate_MPO_sG(sG);

testMPS = deepcopy(initialMPS)
# testMPS = deepcopy(eigenMode)
ξs = Float64[]
sqOps = [] # length 2, 4-leg tensors
eigStates = [] # length 2, 4-leg tensors
coeffs = [] # length 2, 2-leg tensors

for siteIdx in eachindex(testMPS)
    if mod(siteIdx, 2) == 0
        k = abs(siteIdx ÷ 2)
        nMaxk = sG.modeOccupations[2, :][siteIdx]
        ξ = normal_in_interval(-0.15, 0.15)  # optimal range (-0.15, 0.15) for 2 eigenmodes
        # ξ = normal_in_interval(-1e-7, 1e-7)  

        push!(ξs, ξ)
        physSpaceL, physSpaceR = space(testMPS[siteIdx + 0], 2), space(testMPS[siteIdx + 1], 2)
        sqOp = squeezingOp(ξ, nMaxk, -k, k, physSpaceL, physSpaceR)
        push!(sqOps, sqOp)
        D, V = eig(sqOp, (1, 2), (3, 4))
        splitIso = isometry(fuse(physSpaceL, physSpaceR), physSpaceL ⊗ physSpaceR)
        @tensor V[-1 -2; -3 -4] := V[-1, -2, 1] * splitIso[1, -3, -4]

        push!(eigStates, V)
        push!(coeffs, D)

        @tensor localBond[-1 -2 -3; -4] := V[-2, -3, 1, 3] * testMPS[siteIdx + 0][-1, 1, 2] *
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

println("Sampling single tensors...")
testMPS1 = deepcopy(testMPS)
mpsSample, momSample = sample_MPS!(testMPS1)
println("Sample of local basis (index): $(mpsSample)")
println("Sample of local basis (momentum): $(momSample)")


println("Sampling in blocks...")
mpsSample, momSample = sample_MPS_pair!(testMPS, eigStates)
println("Sample of local basis (index): $(mpsSample)")
println("Sample of local basis (momentum): $(momSample)")

# Approach 1: initialize state as eigenstate of squeezing operator
modeSectors = [eigState.colr for eigState in eigStates]
interState = sample_to_BPS(mpsSample, momSample, testMPS, coeffs, modeSectors)
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
blockState = normalizeMPS(blockState)
println("Energy of BPS: $(expectation_value_mpo(blockState, hamMPO))")

# Approach 2: initialize state as CPS in new basis and apply inverse transformation
productState = sample_to_CPS(mpsSample, momSample, testMPS)

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
println("Energy of rotated CPS: $(expectation_value_mpo(productState, hamMPO))")



nothing