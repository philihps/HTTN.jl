using LinearAlgebra
using TensorKit
using HTTN

# sZ_up = [1, 0];
# sZ_down = [0, 1];
# sX_left = 
# sZ_chain = kron(kron(sZ_up, sZ_up), sZ_up);
# sigmaZ = (1/2) * [1 0; 0 -1];
# sigmaZ_chain = kron(kron(sigmaZ, sigmaZ), sigmaZ);
# println(dot(sZ_chain, sigmaZ_chain * sZ_chain))

# rotY = [cos(pi/4) -sin(pi/4); sin(pi/4) cos(pi/4)]
# rotY_chain = kron(kron(rotY, rotY), rotY);
# sX_chain = rotY_chain * sZ_chain # eigenstates of new basis written in old basis
# println(dot(sX_chain, sigmaZ_chain * sX_chain))

# sigmaX_chain = rotY_chain * sigmaZ_chain
# println(dot(sX_chain, sigmaX_chain * sX_chain))

# sigmaX_chain = rotY_chain * sigmaZ_chain * inv(rotY_chain)
# println(dot(sX_chain, sigmaX_chain * sX_chain))


########################################################################
using Random, Distributions

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



println("############Starting from product state############")
mpsSample = [15, 4, 2, 1, 2]
momSample = [0, -3, 1, 0, 2]

finiteCPS = sample_to_CPS(mpsSample, momSample, initialMPS)

numberOperators = local_number_operators(sG);
localOccupations = expectation_values(finiteCPS, numberOperators);
@show localOccupations

# application of Bogoliubov op on mps
testMPS = deepcopy(finiteCPS)
ξs = Float64[]
sqOps = []

for siteIdx in 1 : +1 : (length(testMPS) - 1)
    if mod(siteIdx, 2) == 0
        k = abs(siteIdx ÷ 2)
        nMaxk = sG.modeOccupations[2, :][siteIdx]
        ξ = rand() - 0.5 # uniform random
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
println("############Testing transformation of MPS############")
@show getLinkDimsMPS(testMPS)

println("After Bogoliubov transformation of MPS")
localOccupations = expectation_values(testMPS, numberOperators);
@show localOccupations

# inverse transformation
invtestMPS = deepcopy(testMPS)
for siteIdx in 1 : +1 : (length(invtestMPS) - 1)
    if mod(siteIdx, 2) == 0
        sqOp = sqOps[siteIdx ÷ 2]
        @tensor localBond[-1 -2 -3; -4] := conj(sqOp[1, 3, -2, -3]) * invtestMPS[siteIdx + 0][-1, 1, 2] *
                                           invtestMPS[siteIdx + 1][2, 3, -4] 
        U, S, V, ϵ = tsvd(localBond, (1, 2), (3, 4))

        S /= norm(S)

        U = permute(U, (1, 2), (3,))
        V = permute(S * V, (1, 2), (3,))

        invtestMPS[siteIdx + 0] = U
        invtestMPS[siteIdx + 1] = V
    end
end

invtestMPS = normalizeMPS(invtestMPS)

println("After inverse Bogoliubov transformation of MPS")
@show getLinkDimsMPS(invtestMPS)

localOccupations = expectation_values(invtestMPS, numberOperators);
@show localOccupations

println("############Testing transformation of operator############")
numOps = deepcopy(numberOperators)
# transformation of numberOperators
for siteIdx in 1 : +1 : (length(testMPS) - 1)
    if mod(siteIdx, 2) == 0
        sqOp = sqOps[siteIdx ÷ 2]
        # @tensor localBond[-1 -2; -3 -4] := sqOp[1, 2, -3, -4] * numberOperators[siteIdx + 0][-1, 1] *
        #                                          numberOperators[siteIdx + 1][-2, 2]
        # @tensor test1[-1 -2; -3 -4] := conj(sqOp[1, 2, -1, -2]) * sqOp[1, 2, -3, -4]
        # @tensor test2[-1 -2; -3 -4] := sqOp'[-1, -2, 1, 2] * sqOp[1, 2, -3, -4]

        # @show test1==test2
        @tensor localBond[-1 -2; -3 -4] := conj(sqOp[1, 2, -1, -2]) * numOps[siteIdx + 0][1, 3] *
                                           numOps[siteIdx + 1][2, 4] * sqOp[3, 4, -3, -4]
        U, S, V, ϵ = tsvd(localBond, (1, 3), (2, 4))

        S /= norm(S)
        boundaryL = TensorMap(ones, one(U1Space()), space(S, 1))
        boundaryR = TensorMap(ones, space(S, 1), one(U1Space()))

        U = permute(U * sqrt(S), (1, ), (2, 3))
        @tensor U[-1; -2] := U[-1, -2, 1] * boundaryR[1]
        V = permute(sqrt(S) * V, (1, 2), (3, ))
        @tensor V[-1; -2] := boundaryL[1] * V[1, -1, -2]

        numOps[siteIdx + 0] = U
        numOps[siteIdx + 1] = V
    end
end

localOccupations = expectation_values(testMPS, numOps);
@show localOccupations

println("############Starting from random state############")
println("Perform Bogoliubov transformation of MPS")
testMPS = deepcopy(initialMPS)
ξs = Float64[]
sqOps = []

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
eigStatesSqOps =  []
push!(eigStatesSqOps, undef)

for sqOp in sqOps
    D, V = eig(sqOp, (1, 2), (3, 4))

    # boundaryR = TensorMap(ones, space(D, 1), U1Space())
    # boundaryL = TensorMap(ones, U1Space(), space(D, 1))

    # @tensor eigBasis[-1 -2; -3] := V[-1, -2, 1] * boundaryR[1, -3]
    # @tensor eigBasisInv[-1; -2 -3] := V'[1, -2, -3] * boundaryL[-1, 1]
    @tensor eigBasis[-1 -2; -3] := V[-1, -2, 1] * sqrt(D)[1, -3]
    @tensor eigBasisInv[-1; -2 -3] := sqrt(D)[-1, 1] * V'[1, -2, -3] 

    @tensor eigBasisMink[-1; -2] := eigBasis[-1, 1, 2] * eigBasisInv[2, -2, 1]
    @tensor eigBasisPlusk[-1; -2] := eigBasis[1, -1, 2] * eigBasisInv[2, 1, -2]
    push!(eigStatesSqOps, eigBasisMink)
    push!(eigStatesSqOps, eigBasisPlusk)
end

# define projectors in new basis
projsNew = []
push!(projsNew, undef)
for (i, eigState) in enumerate(eigStatesSqOps)
    if i != 1
        @tensor projVector[-1; -2] := eigState'[-1, 1] * eigState[1, -2]
        # projVector /= norm(projVector)
        # projVector = normalizeMPO(projVector)
        push!(projsNew, projVector)
    end
end

mpsSample, momSample = sample_MPS!(testMPS, projsNew)
println("Sample of local basis (index): $(mpsSample)")
println("Sample of local basis (momentum): $(momSample)")

println("Initialize sampled mps in new basis")
# for siteIdx in 1 : +1 : (length(finiteMPS) - 1)
#     if mod(siteIdx, 2) == 0
#         sqOp = sqOps[siteIdx ÷ 2]
#         @tensor localBond[-1 -2 -3; -4] := sqOp[-2, -3, 1, 3] * finiteMPS[siteIdx + 0][-1, 1, 2] *
#                                            finiteMPS[siteIdx + 1][2, 3, -4]
        
#         U, S, V, ϵ = tsvd(localBond, (1, 2), (3, 4))

#         S /= norm(S)
#         U = permute(U, (1, 2), (3,))
#         V = permute(S * V, (1, 2), (3,))

#         finiteMPS[siteIdx + 0] = U
#         finiteMPS[siteIdx + 1] = V
#     end
# end
finiteMPS = sample_to_CPS_basis(mpsSample, momSample, testMPS, eigStatesSqOps)
finiteMPS = normalizeMPS(finiteMPS)

# inverse transformation
invtestMPS = deepcopy(finiteMPS)
for siteIdx in 1 : +1 : (length(invtestMPS) - 1)
    if mod(siteIdx, 2) == 0
        sqOp = sqOps[siteIdx ÷ 2]
        @tensor localBond[-1 -2 -3; -4] := conj(sqOp[1, 3, -2, -3]) * invtestMPS[siteIdx + 0][-1, 1, 2] *
                                           invtestMPS[siteIdx + 1][2, 3, -4] 
        U, S, V, ϵ = tsvd(localBond, (1, 2), (3, 4))

        S /= norm(S)

        U = permute(U, (1, 2), (3,))
        V = permute(S * V, (1, 2), (3,))

        invtestMPS[siteIdx + 0] = U
        invtestMPS[siteIdx + 1] = V
    end
end

invtestMPS = normalizeMPS(invtestMPS)
println("After inverse Bogoliubov transformation of MPS")
localOccupations = expectation_values(invtestMPS, numberOperators);
@show localOccupations


numOps = deepcopy(numberOperators)
# transformation of numberOperators
for siteIdx in 1 : +1 : (length(finiteMPS) - 1)
    if mod(siteIdx, 2) == 0
        sqOp = sqOps[siteIdx ÷ 2]
        # @tensor localBond[-1 -2; -3 -4] := sqOp[1, 2, -3, -4] * numberOperators[siteIdx + 0][-1, 1] *
        #                                          numberOperators[siteIdx + 1][-2, 2]
        # @tensor test1[-1 -2; -3 -4] := conj(sqOp[1, 2, -1, -2]) * sqOp[1, 2, -3, -4]
        # @tensor test2[-1 -2; -3 -4] := sqOp'[-1, -2, 1, 2] * sqOp[1, 2, -3, -4]

        # @show test1==test2
        @tensor localBond[-1 -2; -3 -4] := conj(sqOp[1, 2, -1, -2]) * numOps[siteIdx + 0][1, 3] *
                                           numOps[siteIdx + 1][2, 4] * sqOp[3, 4, -3, -4]
        U, S, V, ϵ = tsvd(localBond, (1, 3), (2, 4))

        S /= norm(S)
        boundaryL = TensorMap(ones, one(U1Space()), space(S, 1))
        boundaryR = TensorMap(ones, space(S, 1), one(U1Space()))

        U = permute(U * sqrt(S), (1, ), (2, 3))
        @tensor U[-1; -2] := U[-1, -2, 1] * boundaryR[1]
        V = permute(sqrt(S) * V, (1, 2), (3, ))
        @tensor V[-1; -2] := boundaryL[1] * V[1, -1, -2]

        numOps[siteIdx + 0] = U
        numOps[siteIdx + 1] = V
    end
end
println("Apply transformed op on transformed MPS")

localOccupations = expectation_values(finiteMPS, numOps);
@show localOccupations


nothing

########################################################################
# Initiate CPS in new basis?
# 1. Initiate CPS in old basis
# 2. Apply Bogoliubov transformation onto CPS
# Another way to split numOp. Similar to proj?