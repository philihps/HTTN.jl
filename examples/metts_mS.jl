using HTTN
using TensorKit
using JLD2

OUTPUT_PATH = "/home/"

# set modelName
modelName = "massiveSchwinger"

# set display parameters
modeOrdering = true;

# set truncation parameters
truncMethod = 5;
kMax = 1;
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
display(mS.modeOccupations)
hamMPO = generate_MPO_mS(mS)

# construct physical and virtual vector spaces for the MPS
boundarySpaceL = U1Space(0 => 1);
boundarySpaceR = U1Space(0 => 1);
physSpaces = mS.physSpaces;
virtSpaces = constructVirtSpaces(mS.physSpaces, boundarySpaceL, boundarySpaceR;
                                 removeDegeneracy = true);

# initialize random MPS
initialTensors = Vector{TensorMap{ComplexF64}}(undef, length(physSpaces));
for siteIdx in eachindex(physSpaces)
    physSpace = physSpaces[siteIdx]
    initialTensors[siteIdx] = randn(ComplexF64, virtSpaces[siteIdx] ⊗ physSpace,
                                    virtSpaces[siteIdx + 1])
end
initialMPS = SparseMPS(initialTensors; normalizeMPS = true);

numTimeStep = 500 # 1000
numMETTS = 100
inverseT = 1000
SLURM_ARRAY_JOB_ID = "test"
FILE_INFO = "$(modelName)_m_$(fermionMass)_invT_$(inverseT)_$(SLURM_ARRAY_JOB_ID)"

println("System info: kMax=$kMax, nMax=$nMax, nMaxZM=$nMaxZM, truncMethod=$truncMethod, theta=$θ, M=$M, L=$L, fermionMass=$fermionMass")
warmup_energies, energies, truncErrs, totalNumMETTS = metts_basis(initialMPS, hamMPO,
                                                                  mS, numTimeStep,
                                                                  inverseT;
                                                                  alg = METTS2(;
                                                                         numWarmUp = 3,
                                                                         numMETTS = numMETTS,
                                                                         numMETTSMin = 3,
                                                                         doBasisExtend = false,
                                                                         sqZero = false,
                                                                         transfRange = (-0.2, 0.2),
                                                                         tol = 1.0))

@save OUTPUT_PATH * FILE_INFO * "_warmup_energies.jld2" warmup_energies
@save OUTPUT_PATH * FILE_INFO * "_energies.jld2" energies
@save OUTPUT_PATH * FILE_INFO * "_trunc_err.jld2" truncErrs
@save OUTPUT_PATH * FILE_INFO * "_numMETTS.jld2" totalNumMETTS
