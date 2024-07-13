
module HTTN

    # include external packages
    using Base: @kwdef

    import LinearAlgebra
    using LinearAlgebra: diag, diagm

    using FFTW
    using JLD
    using KrylovKit
    using OptimKit
    using Printf
    using LaTeXStrings
    using Plots
    using Roots
    using SpecialFunctions
    using TensorKit
    using Zygote

    # export states
    export SparseMPS

    # export operators
    export SparseMPO
    export SparseEXP

    # export environments
    export SparseENV

    # export models
    export SineGordonParameters
    export SineGordonModel
    export MassiveSchwingerParameters
    export MassiveSchwingerModel
    export getMomentumModes
    export initializeVacuumMPS, initializeMPS
    export generate_H0, generate_H1, generate_MPO_mS, generate_MPO_sG
    export updateBogoliubovPrameters
    export local_number_operators
    export pairing_operators

    # export algorithms
    export find_groundstate, find_excitedstate
    export DMRG2, DMRG2BO
    export perform_timestep
    export TDVP2
    export metts
    export METTS2

    # export mpo compression functions
    export compress_MPO, convert_MPO_33block, convert_33block_MPO

    # export expectation value functions
    export expectation_value_mpo, expectation_values
    export compute_entanglement_spectra, compute_entanglement_entropies, compute_arbitrary_bipartition, compute_mutual_information, compute_mutual_information_pairs
    export compute_phase_distribution

    # export utility functions
    export constructPhysSpaces, constructVirtSpaces, getLinkDimsMPS, getLinkDimsMPO
    export save_to_file, load_from_file


    # include source files
    include("states/SparseMPS.jl")
    include("operators/SparseMPO.jl")
    include("environments/utils_environments.jl")

    include("models/qft_models.jl")
    include("models/mS.jl")
    include("models/sG.jl")
    include("models/utils.jl")

    include("algorithms/dmrg.jl")
    include("algorithms/tdvp.jl")
    include("algorithms/metts.jl")
    include("algorithms/basis_optimization.jl")
    include("algorithms/expectation_values.jl")
    include("algorithms/entanglement_quantities.jl")
    include("algorithms/full_counting_statistics.jl")
    
    include("algorithms/mpo_compression.jl")

    include("utility/bosonOperators.jl")
    include("utility/vectorSpaces.jl")

end