#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

Pkg.activate(".")
using HTTN
using LaTeXStrings
using Plots
using Printf
using TensorKit

let

    # set truncation parameters
    modelName = "coupledRotors"
    BC = "NBC" # "NBC, "DBC", or "PBC"
    M = 4 # number of coupled rotors
    N = 10 # maximal eigenvalue of momentum operator

    # set model parameters
    β = 1.0
    ωPre = sqrt(1.0)
    ωPost = 0.0
    κ = 2.0

    # set DMRG and TDVP truncation parameters
    bondDim = 64
    truncErrD = 1e-6
    truncErrT = 1e-6

    # set time step and total time for time evolution
    δT = 1e-1
    totalTime = 5.0
    numTimeSteps = Int(totalTime / δT)

    # create NamedTuple for truncation parameters
    truncationParameters = (M = M, N = N)

    # create NamedTuple for model parameters
    hamiltonianParameters_Pre = (β = β, ω = ωPre, κ = κ, BC = BC)
    hamiltonianParameters_Post = (β = β, ω = ωPost, κ = κ, BC = BC)
    
    # construct coupled rotors model
    cR_Pre = CoupledRotorsModel(truncationParameters, hamiltonianParameters_Pre)
    cR_Post = CoupledRotorsModel(truncationParameters, hamiltonianParameters_Post)
    modeOccupations = cR_Pre.modeOccupations
    physSpaces = cR_Pre.physSpaces
    display(modeOccupations)
    display(reshape(physSpaces, 1, :))
    println()

    # construct pre-quench MPO
    preQuenchMPO = generate_MPO_cR(cR_Pre)
    println("MPO bond dimensions pre-quench:")
    println(getLinkDimsMPO(preQuenchMPO), "\n")

    # construct post-quench MPO
    postQuenchMPO = generate_MPO_cR(cR_Post)
    println("MPO bond dimensions post-quench (ω = 0):")
    println(getLinkDimsMPO(postQuenchMPO), "\n")


    # -------------------------------------------------------------
    # DMRG

    # initialize vaccum MPS (ground state of non-interacting Hamiltonian)
    vacuumMPS = initializeVacuumMPS(cR_Pre)
    initialMPS = initializeMPS(cR_Pre, vacuumMPS)

    # run DMRG to compute pre-quench ground state
    groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, preQuenchMPO,
                                                                    DMRG2(; bondDim = bondDim,
                                                                            truncErr = truncErrD,
                                                                            maxIterationsInit = 10,
                                                                            maxIterations = 2,
                                                                            subspaceExpansion = false,
                                                                            verbosePrint = 1))
    @printf("ground state energy E0 pre-quench = %0.8f\n\n", groundStateEnergy)


    # -------------------------------------------------------------
    # TDVP

    # copy input state
    timeEvolvedMPS = copy(groundStateMPS)

    # initialize array to store entanglement entropy
    storeEntanglementEntropy = zeros(Float64, 0, length(physSpaces) - 1)

    # initialize array to store bond dimension
    storeBondDimension = zeros(Float64, 0, length(physSpaces) - 1)

    # initialize array to store maximal truncation error
    storeTruncationError = zeros(Float64, 0, 2)

    # # initialize array to store energy expectation value
    # storeEnergy = zeros(Float64, 0, 2)

    # run TDVP to evolve the pre-quench ground state with the post-quench Hamiltonian
    ϵ = 0.0
    for timeStep in 0:numTimeSteps

        @printf("timeStep = %d/%d\n", timeStep, numTimeSteps)

        # perform time step
        if timeStep > 0
            timeEvolvedMPS, envL, envR, ϵ = perform_timestep!(timeEvolvedMPS, postQuenchMPO, δT,
                                                            TDVP2(; bondDim = bondDim,
                                                                    truncErrT = truncErrT,
                                                                    verbosePrint = 1,
                                                                    extendBasis = false,))
        end

        # compute entanglement entropies
        mpsEntanglementEntropies = compute_entanglement_entropies(timeEvolvedMPS)
        storeEntanglementEntropy = vcat(storeEntanglementEntropy, reshape(mpsEntanglementEntropies, 1, :))

        # get maximal bond dimension
        virtBondDims = getLinkDimsMPS(timeEvolvedMPS)
        storeBondDimension = vcat(storeBondDimension, reshape(virtBondDims[2:(end - 1)], 1, :))

        # store truncation error
        storeTruncationError = vcat(storeTruncationError, [δT * timeStep ϵ])

        # # compute energy of time-evolved state
        # energyExpectationValue = expectation_value_mpo(timeEvolvedMPS, postQuenchMPO)
        # storeEnergy = vcat(storeEnergy, [δT * timeStep real(energyExpectationValue)])

    end

    display(storeEntanglementEntropy)
    display(storeBondDimension)
    # display(storeEnergy)


    # plot for the entanglement entropy over time
    plotEntanglementEntropy = plot(; xlabel = L"t", ylabel = L"S(t)", frame = :box)
    for idxB in axes(storeEntanglementEntropy, 2)
        plot!(plotEntanglementEntropy,
            δT * collect(0:numTimeSteps),
            storeEntanglementEntropy[:, idxB];
            ylims = (0, Inf),
            linewidth = 2.0,
            label = "",)
    end
    display(plotEntanglementEntropy)


    # plot for the bond dimension over time
    plotBondDimensions = plot(; xlabel = L"t", ylabel = L"\chi(t)", frame = :box)
    for idxB in axes(storeBondDimension, 2)
        plot!(plotBondDimensions,
            δT * collect(0:numTimeSteps),
            storeBondDimension[:, idxB];
            linewidth = 2.0,
            label = "",)
    end
    display(plotBondDimensions)


    # plot for the truncation error over time
    plotTruncationError = plot(; xlabel = L"t", ylabel = L"\varepsilon(t)", frame = :box)
    plot!(plotTruncationError,
            storeTruncationError[:, 1],
            storeTruncationError[:, 2];
            linewidth = 2.0,
            label = "",)
    display(plotTruncationError)



    # # plot for the energy over time
    # plotEnergy = plot(; xlabel = L"t", ylabel = L"E(t)", frame = :box)
    # plot!(plotEnergy,
    #         storeEnergy[:, 1],
    #         storeEnergy[:, 2];
    #         linewidth = 2.0,
    #         label = "",)
    # display(plotEnergy)

end