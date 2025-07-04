#!/usr/bin/env julia

# clear console
Base.run(`clear`)

using Pkg
using Revise

Pkg.activate(".")
using HTTN
using KrylovKit
using LaTeXStrings
using Interpolations
using Roots
using Plots
using Printf
# using IntervalRootFinding
# using IntervalArithmetic
# using IntervalArithmetic.Symbols
using TensorKit

# let

    # set truncation parameters
    modelName = "coupledRotors"
    BC = "NBC" # "NBC, "DBC", or "PBC"
    M = 2 # number of coupled rotors
    N = 10 # maximal eigenvalue of momentum operator

    # set model parameters
    β = 1.0
    ωPre = sqrt(1.5)
    ωPost = 0.0
    κ = 10.0

    # set DMRG and TDVP truncation parameters
    bondDim = 128
    truncErrD = 1e-6
    truncErrT = 1e-6

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

    # construct center-of-mass Hamiltonian (P) and relative Hamiltonian (M) MPO
    mpoP = generate_H0_CM(cR_Post)
    mpoM = generate_MPO_cR(cR_Post) - mpoP


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

    # # initialize vaccum MPS (ground state of non-interacting Hamiltonian)
    # vacuumMPS = initializeVacuumMPS(cR_Post)
    # initialMPS = initializeMPS(cR_Post, vacuumMPS)

    # # run DMRG to compute post-quench ground state
    # groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, postQuenchMPO,
    #                                                                 DMRG2(; bondDim = bondDim,
    #                                                                         truncErr = truncErrD,
    #                                                                         maxIterationsInit = 20,
    #                                                                         maxIterations = 2,
    #                                                                         subspaceExpansion = false,
    #                                                                         verbosePrint = 1))
    # @printf("ground state energy E0 post-quench = %0.8f\n\n", groundStateEnergy)


    # # -------------------------------------------------------------
    # # thermal state cooling

    # # set temperature and final temperature for imaginary time evolution
    # δβ = 5e-2
    # β = 5e0
    # numCoolingSteps = Int(β / δβ)

    # # create initial thermal density matrix
    # thermalDensityMatrix = constructIdentiyMPO(physSpaces, ComplexSpace(1))
    # thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)

    # # create infinitesimal thermal density matrix
    # infinitesimalTDM = initializeThermalDensityMatrix(, δβ = δβ, expansionOrder = 3, truncErr = 1e-6, maxDim = 128)

    # # initialize array to store entanglement entropy
    # storeEntanglementEntropy = zeros(Float64, 0, length(physSpaces) - 1)

    # # initialize array to store bond dimension
    # storeBondDimension = zeros(Float64, 0, 1 + length(physSpaces) - 1)

    # # initialize array to store maximal truncation error
    # storeTruncationError = zeros(Float64, 0, 2)

    # # initialize array to store energy expectation value
    # storeEnergy = zeros(Float64, 0, 2)

    # # run TDVP to evolve the pre-quench ground state with the post-quench Hamiltonian
    # ϵ = 0.0
    # for timeStep in 0:numCoolingSteps

    #     @printf("timeStep = %d/%d\n", timeStep, numCoolingSteps)

    #     # perform cooling step
    #     if timeStep > 0
    #         thermalDensityMatrix = multiplyMPOs(thermalDensityMatrix, infinitesimalTDM, truncErr = 1e-6, maxDim = 128)
    #         thermalDensityMatrix = normalizeMPO(thermalDensityMatrix)
    #     end

    #     # # compute entanglement entropies
    #     # mpsEntanglementEntropies = compute_entanglement_entropies(thermalDensityMatrix)
    #     # storeEntanglementEntropy = vcat(storeEntanglementEntropy, reshape(mpsEntanglementEntropies, 1, :))

    #     # get maximal bond dimension
    #     virtBondDims = getLinkDimsMPO(thermalDensityMatrix)
    #     storeBondDimension = vcat(storeBondDimension, [δβ * timeStep reshape(virtBondDims[2:(end - 1)], 1, :)])

    #     # # store truncation error
    #     # storeTruncationError = vcat(storeTruncationError, [δβ * timeStep ϵ])

    #     # compute energy of cooled state
    #     energyExpectationValue = expectation_values_density_matrix(thermalDensityMatrix, postQuenchMPO)
    #     storeEnergy = vcat(storeEnergy, [δβ * timeStep real(energyExpectationValue)])

    # end

    # display(storeBondDimension)

    # # plot for the bond dimension over time
    # plotBondDimensions = plot(; xlabel = L"T", ylabel = L"\chi(T)", frame = :box)
    # for idxB in axes(storeBondDimension, 2)
    #     if idxB > 1
    #         plot!(plotBondDimensions,
    #             1 ./ storeBondDimension[:, 1],
    #             storeBondDimension[:, idxB];
    #             linewidth = 2.0,
    #             label = "",)
    #     end
    # end
    # display(plotBondDimensions)

    # display(storeEnergy)

    # # plot for the energy over time
    # plotEnergy = plot(; xaxis = :log10, xlabel = L"T", ylabel = L"E(T)", frame = :box)
    # plot!(plotEnergy,
    #         1 ./ storeEnergy[:, 1],
    #         storeEnergy[:, 2];
    #         linewidth = 2.0,
    #         label = "",)
    # plot!(plotEnergy,
    #         1 ./ storeEnergy[:, 1],
    #         fill(groundStateEnergy, length(storeEnergy[:, 2]));
    #         linewidth = 2.0,
    #         linestyle = :dash,
    #         label = L"E_0",)
    # display(plotEnergy)


    # -------------------------------------------------------------
    # GGE

    # compute energy of initial state for both P and M Hamiltonian
    targetEnergyP = real(expectation_value_mpo(groundStateMPS, mpoP))
    targetEnergyM = real(expectation_value_mpo(groundStateMPS, mpoM))
    println("target energy P = ", targetEnergyP)
    println("target energy M = ", targetEnergyM)

    # set infinitesimal temperature step and maximal inverse temperature
    δβ = 5e-3


    # compute thermal density matrix for mpoP
    tdm_P, λP, thermalEnergiesP = produceThermalState(
        mpoP,
        targetEnergyP;
        δβ = δβ,
        truncErr= 1e-8,
        maxDim = 128,
        convTol = 1e-3,
        verbosePrint = 0,
    )
    println("optimal λP = ", λP)

    # plot for the energy over time
    plotEnergyP = plot(; xaxis = :identity, xlabel = L"T", ylabel = L"E(T)", frame = :box)
    plot!(plotEnergyP,
        1 .* thermalEnergiesP[:, 1],
        thermalEnergiesP[:, 2];
        linewidth = 2.0,
        markers = :circle,
        label = "MPO data",
    )
    display(plotEnergyP)


    # compute thermal density matrix for mpoM
    tdm_M, λM, thermalEnergiesM = produceThermalState(
        mpoM, 
        targetEnergyM;
        δβ = δβ,
        truncErr= 1e-8,
        maxDim = 128,
        convTol = 1e-2,
        verbosePrint = 0,
    )
    println("optimal λM = ", λM)

    # plot for the energy over time
    plotEnergyM = plot(; xaxis = :identity, xlabel = L"T", ylabel = L"E(T)", frame = :box)
    plot!(plotEnergyM,
        1 .* thermalEnergiesM[:, 1],
        thermalEnergiesM[:, 2];
        linewidth = 2.0,
        markers = :circle,
        label = "MPO data",
    )
    display(plotEnergyM)

    # compute final thermal density matrix
    tdm_GGE = multiplyMPOs(tdm_P, tdm_M, truncErr = 1e-8, maxDim = 128)
    tdm_GGE = normalizeMPO(tdm_GGE)

    # compute half-system von Neumann entropy
    rdm_HS = real(reducedDensityMatrixHalfSystem(tdm_GGE))
    SvN = -real(tr(rdm_HS * log(rdm_HS)))
    println("von Neumann entropy S = ", SvN)

# end