using HTTN
using TensorKit
using JLD2
using Revise

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
vacuumMPS = initializeVacuumMPS(mS; modeOrdering = modeOrdering)
initialMPS = initializeMPS(mS, vacuumMPS; modeOrdering = modeOrdering)

numTimeStep = 500 # 1000
numMETTS = 100
inverseT = 50
FILE_INFO = "$(modelName)_m_$(fermionMass)_invT_$(inverseT)_$(SLURM_ARRAY_JOB_ID)"


println("System info: kMax=$kMax, nMax=$nMax, nMaxZM=$nMaxZM, truncMethod=$truncMethod, theta=$θ, M=$M, L=$L, fermionMass=$fermionMass")
warmup_energies, energies, truncErrs, totalNumMETTS = metts_basis(initialMPS, hamMPO,
                                                                    mS, numTimeStep,
                                                                    inverseT,
                                                                    METTS2(;
                                                                            numWarmUp = 50,
                                                                            numMETTS = numMETTS,
                                                                            doBasisExtend = true,
                                                                            tol = 1.0))

@save OUTPUT_PATH * FILE_INFO * "_warmup_energies.jld2" warmup_energies
@save OUTPUT_PATH * FILE_INFO * "_energies.jld2" energies
@save OUTPUT_PATH * FILE_INFO * "_trunc_err.jld2" truncErrs
@save OUTPUT_PATH * FILE_INFO * "_numMETTS.jld2" totalNumMETTS
