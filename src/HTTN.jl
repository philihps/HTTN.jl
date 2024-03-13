
module HTTN

# include external packages
using Base: @kwdef

import LinearAlgebra
using LinearAlgebra: diag, diagm

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

# export algorithms
export find_groundstate, find_excitedstate
export DMRG2
export perform_timestep
export TDVP2

# export utility functions
export constructPhysSpaces, constructVirtSpaces
export save_to_file, load_from_file


# include source files
include("states/SparseMPS.jl")
include("operators/SparseMPO.jl")
include("environments/utils_environments.jl")
include("models/QFTModels.jl")

include("algorithms/dmrg.jl")
include("algorithms/expectationvalues.jl")
include("algorithms/tdvp.jl")

include("utility/bosonOperators.jl")
include("utility/vectorSpaces.jl")


end