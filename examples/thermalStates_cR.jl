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
    M = 4 # number of coupled rotors
    N = 2 # maximal eigenvalue of momentum operator

    # set model parameters
    β = 1.0
    ωPre = sqrt(1.0)
    ωPost = 0.0
    κ = 2.0

    # set λP and λM for GGE
    λP = 1.0
    λM = 1.0

    # set DMRG and TDVP truncation parameters
    bondDim = 64
    truncErrD = 1e-6
    truncErrT = 1e-6

    # create NamedTuple for truncation parameters
    truncationParameters = (M = M, N = N)

    # create NamedTuple for model parameters
    hamiltonianParameters_Pre = (β = β, ω = ωPre, κ = κ, BC = BC, λP = λP, λM = λM)
    hamiltonianParameters_Post = (β = β, ω = ωPost, κ = κ, BC = BC, λP = λP, λM = λM)
    
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

    # # construct post-quench MPO
    # postQuenchMPO = generate_MPO_cR_CM_REL(cR_Post)
    # println("MPO bond dimensions post-quench (ω = 0):")
    # println(getLinkDimsMPO(postQuenchMPO), "\n")

    # -------------------------------------------------------------
    # DMRG

    # initialize vaccum MPS (ground state of non-interacting Hamiltonian)
    vacuumMPS = initializeVacuumMPS(cR_Post)
    initialMPS = initializeMPS(cR_Post, vacuumMPS)

    # run DMRG to compute pre-quench ground state
    groundStateMPS, groundStateEnergy, truncErrors = find_groundstate(initialMPS, preQuenchMPO,
                                                                    DMRG2(; bondDim = bondDim,
                                                                            truncErr = truncErrD,
                                                                            maxIterationsInit = 10,
                                                                            maxIterations = 2,
                                                                            subspaceExpansion = false,
                                                                            verbosePrint = 1))
    @printf("ground state energy E0 post-quench = %0.8f\n\n", groundStateEnergy)


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

    # set infinitesimal temperature step and maximal inverse temperature
    δβ = 1e-2
    β = 5.0
    
    # construct center-of-mass Hamiltonian (P) and relative Hamiltonian (M) MPO
    mpoP = generate_H0_CM(cR_Post)
    mpoM = generate_MPO_cR(cR_Post) + (-1 * mpoP)

    # compute energy of initial state for both P and M Hamiltonian
    targetEnergyP = real(expectation_value_mpo(groundStateMPS, mpoP))
    targetEnergyM = real(expectation_value_mpo(groundStateMPS, mpoM))
    println("target energy P = ", targetEnergyP)
    println("target energy M = ", targetEnergyM)

    # cooling process for mpoP
    _, thermalEnergyP = produceThermalState(
        mpoP, β; δβ = δβ, verbosePrint = 1
    )

    # interpolate thermal state energy
    linear_interpolP = linear_interpolation(thermalEnergyP[:, 1], thermalEnergyP[:, 2])

    # plot for the energy over time
    plotEnergyP = plot(; xaxis = :identity, xlabel = L"T", ylabel = L"E(T)", frame = :box)
    plot!(plotEnergyP,
        1 .* thermalEnergyP[:, 1],
        thermalEnergyP[:, 2];
        linewidth = 2.0,
        markers = :circle,
        label = "MPO data",
    )

    plot!(plotEnergyP,
        1 .* thermalEnergyP[:, 1],
        linear_interpolP.(1 .* thermalEnergyP[:, 1],),
        linewidth = 2.0,
        linestyle = :dot,
        label = "linear interp."
    )

    display(plotEnergyP)

    # define function to compute root of
    rootFuncP(β::Float64) = linear_interpolP(β) - targetEnergyP
    λP = find_zero(rootFuncP, (0.0, β))


    # cooling process for mpoM
    _, thermalEnergyM = produceThermalState(
        mpoM, β; δβ = δβ, verbosePrint = 1
    )

    # interpolate thermal state energy
    linear_interpolM = linear_interpolation(thermalEnergyM[:, 1], thermalEnergyM[:, 2])

    # plot for the energy over time
    plotEnergyM = plot(; xaxis = :identity, xlabel = L"T", ylabel = L"E(T)", frame = :box)
    plot!(plotEnergyM,
        1 .* thermalEnergyM[:, 1],
        thermalEnergyM[:, 2];
        linewidth = 2.0,
        markers = :circle,
        label = "MPO data",
    )

    plot!(plotEnergyM,
        1 .* thermalEnergyM[:, 1],
        linear_interpolM.(1 .* thermalEnergyM[:, 1],),
        linewidth = 2.0,
        linestyle = :dot,
        label = "linear interp."
    )
    display(plotEnergyM)

    # define function to compute root of
    rootFuncM(β::Float64) = linear_interpolM(β) - targetEnergyM
    λM = find_zero(rootFuncM, (0.0, β))

    # generate thermal density matrix for mpoP
    tdm_P, thermalEnergyP = produceThermalState(
        mpoP, λP; δβ = δβ, verbosePrint = 1
    )

    # generate thermal density matrix for mpoM
    tdm_M, thermalEnergyM = produceThermalState(
        mpoM, λM; δβ = δβ, verbosePrint = 1
    )

    # compute final thermal density matrix
    tdm_GGE = multiplyMPOs(tdm_P, tdm_M, truncErr = 1e-6, maxDim = 256)
    tdm_GGE = real(normalizeMPO(tdm_GGE))

    # compute half-system von Neumann entropy
    rdm_HS = reducedDensityMatrixHalfSystem(mpoP)
    D, V = eig(rdm_HS)
    SvN = -real(tr(D * log(D)))
    println("von Neumann entropy S = ", SvN)

# end