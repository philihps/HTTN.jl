
__precompile__(false)
module HTTN

# include external packages
using Base: @kwdef

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: diag, diagm

using FFTW
using JLD2
using KrylovKit
using OptimKit
using Printf
using LaTeXStrings
using Roots
using SpecialFunctions
using TensorKit
using Zygote

import Distributions: Normal

using PackageExtensionCompat

# export states
export SparseMPS, normMPS

# export operators
export SparseMPO

# export models
export SineGordonParameters
export SineGordonModel
export MassiveSchwingerParameters
export MassiveSchwingerModel
export CoupledRotorsParameters
export CoupledRotorsModel
export initializeVacuumMPS, initializeMPS
export generate_H0, generate_H1, generate_MPO_mS, generate_MPO_sG, generate_MPO_cR,
       generate_H0_CM
export updateBogoliubovParameters
export local_number_operators
export pairing_operators
export getMomentumModes

# export algorithms
export find_groundstate, find_excitedstate
export DMRG2, DMRG2BO
export perform_timestep!
export perform_basisOptimization!
export TDVP2, TDVP2BO
export metts, metts_ZM, transform_basis!
export METTS2

# export mps2vec and mpo2mat functions
export mps2vec, mpo2mat

# export mpo compression functions
export compress_MPO, convert_MPO_33block, convert_33block_MPO

# export expectation value functions
export expectation_value_mpo, expectation_values, expectation_values_density_matrix
export compute_entanglement_spectra,
       compute_entanglement_entropies,
       compute_arbitrary_bipartition,
       compute_mutual_information
export compute_phase_distribution

# export thermal state functions
export initializeThermalDensityMatrix, expectation_values_density_matrix
export constructIdentiyMPO
export produceThermalState, costFunctionGGE, reducedDensityMatrixHalfSystem

# export utility functions
export normalizeMPS, dotMPS, normalizeMPO, dotMPO, trMPO, multiplyMPOs
export constructPhysSpaces, constructVirtSpaces, getLinkDimsMPS, getLinkDimsMPO
export diagTM
export transform_basis!, sample_MPS!, sample_MPS_block!, sample_to_BPS, sample_to_CPS

# export save_to_file, load_from_file
export compute_average

# export Bogoliubov transformation functions
export squeezingOp, singleSqueezingOp

# include source files
include("utility/bosonOperators.jl")
include("utility/shared.jl")
include("utility/vectorSpaces.jl")
include("utility/SparseMPS.jl")
include("utility/SparseMPO.jl")
include("utility/environments.jl")
include("utility/statAnalysis.jl")

include("models/qft_models.jl")
include("models/mS_sG.jl")
include("models/coupled_rotors.jl")
include("models/utils.jl")

include("algorithms/dmrg.jl")
include("algorithms/tdvp.jl")
include("algorithms/metts.jl")
include("algorithms/basis_optimization.jl")
include("algorithms/expectation_values.jl")
include("algorithms/entanglement_quantities.jl")
include("algorithms/full_counting_statistics.jl")
include("algorithms/mpo_compression.jl")
include("algorithms/thermal_states.jl")

end
