
module HTTN

# include external packages
using Base: @kwdef

import LinearAlgebra
using LinearAlgebra: diag, diagm

using FFTW
using JLD
using KrylovKit
using Printf
using SpecialFunctions
using TensorKit

# export states
export SparseMPS

# export operators
export SparseMPO
export SparseEXP

# export environments
export SparseENV

# export models
export MassiveSchwingerParameters
export MassiveSchwingerModel
export local_number_operators

# export algorithms
export find_groundstate, find_excitedstate
export DMRG2
export perform_timestep
export TDVP2

# export expectation value functions
export expectation_value_mpo, expectation_values
export compute_entanglement_spectra, compute_entanglement_entropies
export compute_phase_distribution

# export utility functions
export constructPhysSpaces, constructVirtSpaces
export save_to_file, load_from_file


# include source files
include("states/SparseMPS.jl")
include("operators/SparseMPO.jl")
include("environments/utils_environments.jl")

include("models/qft_models.jl")
include("models/mS.jl")

include("algorithms/dmrg.jl")
include("algorithms/tdvp.jl")
include("algorithms/expectation_values.jl")
include("algorithms/entanglement_quantities.jl")
include("algorithms/full_counting_statistics.jl")

include("utility/bosonOperators.jl")
include("utility/vectorSpaces.jl")


end