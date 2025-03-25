#!/usr/bin/env julia

#------------------------------------------------------
# MPO types
#------------------------------------------------------

abstract type AbstractMPO end
abstract type MomentumMPO end
abstract type AbstractFiniteMPO <: AbstractMPO end
abstract type FiniteMomentumMPO <: MomentumMPO end


#--------------------------------------------------------------
# MPO constructors
#--------------------------------------------------------------
struct SparseMomentumMPO{A<:AbstractTensorMap{ComplexF64}} <: FiniteMomentumMPO
    """
    3-index MPO with additional δ-index for momentum transfer between the bra and ket index.
    """
    momentumTensors::Vector{A}

    function MomentumMPO{A}(momentumTensors::Vector{A}) where {A<:AbstractTensorMap{ComplexF64}}
        return new{A}(momentumTensors)
    end

    function MomentumMPO(momentumTensors::Vector{A}) where {A<:AbstractTensorMap{ComplexF64}}
        return new{A}(momentumTensors)
    end
end

struct SparseMPO{A<:AbstractTensorMap{ComplexF64}} <: AbstractFiniteMPO
    """
    4-/2-index MPO
    """
    mpoTensors::Vector{A}

    function SparseMPO{A}(mpoTensors::Vector{A}) where {A<:AbstractTensorMap{ComplexF64}}
        return new{A}(mpoTensors)
    end

    function SparseMPO(mpoTensors::Vector{A}) where {A<:AbstractTensorMap{ComplexF64}}
        return new{A}(mpoTensors)
    end
end



#--------------------------------------------------------------
# SparseMPO utilities
#--------------------------------------------------------------

"""
getKroneckerDeltaSpace(E::MomentumMPO, siteIdx::Int)

Returns the Kronecker-Delta space of the EXP tensor at site 'siteIdx'.

"""
# function getKroneckerDeltaSpace end

function getKroneckerDeltaSpace(E::SparseMPO, siteIdx::Integer)
    return dual(space(E.expTensors[siteIdx], 3))
end

"""
getPhysicalSpace(E::MomentumMPO, siteIdx::Int)

Returns the physical space of the EXP tensor at site 'siteIdx'.

"""
# function getPhysicalSpace end

function getPhysicalSpace(E::SparseMPO, siteIdx::Integer)
    return space(E.expTensors[siteIdx], 1)
end

Base.getindex(E::SparseMPO, idx) = E.expTensors[idx];
Base.size(E::SparseMPO, args...) = size(E.expTensors, args...);
Base.length(E::SparseMPO) = length(E.expTensors);
Base.iterate(E::SparseMPO, args...) = iterate(E.expTensors, args...);

function getLinkDimsMPO(M::SparseMPO)
    return dim.(vcat([getVirtualSpaceL(M, idx) for idx in 1:length(M)],
                     getVirtualSpaceR(M, length(M))))
end
maxLinkDimsMPO(M::SparseMPO) = maximum(getLinkDimsMPO(M))

#--------------------------------------------------------------
# SparseMPO utilities
#--------------------------------------------------------------

"""
getVirtualSpaceL(M::AbstractMPO, siteIdx::Int)

Returns the left virtual space of the MPO tensor at site 'siteIdx'.
This is equivalent to the right virtual space of the MPO tensor at site 'siteIdx - 1'.

"""

function getVirtualSpaceL(M::SparseMPO, siteIdx::Integer)
    return space(M.mpoTensors[siteIdx], 1)
end

"""
getVirtualSpaceR(M::AbstractMPS, siteIdx::Int)
    
Returns the right virtual space of the MPS tensor at site 'siteIdx'.
This is equivalent to the left virtual space of the MPS tensor at site 'siteIdx + 1'.

"""

function getVirtualSpaceR(M::SparseMPO, siteIdx::Integer)
    return dual(space(M.mpoTensors[siteIdx], 3))
end

"""
getPhysicalSpace(M::AbstractMPS, siteIdx::Int)
    
Returns the physical space of the MPS tensor at site 'siteIdx'.

"""
# function getPhysicalSpace end

function getPhysicalSpace(M::SparseMPO, siteIdx::Integer)
    return space(M.mpoTensors[siteIdx], 2)
end

Base.getindex(M::SparseMPO, idx) = M.mpoTensors[idx];
Base.setindex!(M::SparseMPO, mpoTensor, idx::Int) = (M.mpoTensors[idx] = mpoTensor)
Base.size(M::SparseMPO, args...) = size(M.mpoTensors, args...)
Base.length(M::SparseMPO) = length(M.mpoTensors)
Base.eltype(M::SparseMPO) = eltype(M.mpoTensors)
Base.iterate(M::SparseMPO, args...) = iterate(M.mpoTensors, args...)
Base.eachindex(M::SparseMPO, args...) = eachindex(M.mpoTensors, args...)
Base.lastindex(M::SparseMPO) = lastindex(M.mpoTensors)
Base.copy(M::SparseMPO) = SparseMPO(copy(M.mpoTensors))
LinearAlgebra.real(M::SparseMPO) = SparseMPO(real.(M.mpoTensors))
function Base.similar(M::SparseMPO{A}) where {A}
    return SparseMPO{A}(similar(M.mpoTensors))
end

# elementary operations
Base.:-(a::SparseMPO) = -one(scalartype(a)) * a;

# addition
function Base.:+(mpoA::H, mpoB::H) where {H<:SparseMPO}

    # get length of MPOs
    NA = length(mpoA)
    NB = length(mpoB)
    NA != NB &&
        throw(DimensionMismatch("lengths of MPO A ($NA) and MPO B ($NB) do not match"))

    # add MPOs from left to right
    MPOC = Vector{TensorMap{ComplexF64}}(undef, NA)
    if NA == 1

        # combine single tensor
        idxMPO = 1
        MPOC[idxMPO] = mpoA[idxMPO] + mpoB[idxMPO]

    else

        # left boundary tensor
        idxMPO = 1
        isoRA = isometry(space(mpoA[idxMPO], 3)' ⊕ space(mpoB[idxMPO], 3)',
                         space(mpoA[idxMPO], 3)')'
        isoRB = rightnull(isoRA)
        @tensor newTensor[-1 -2; -3 -4] := mpoA[idxMPO][-1, -2, 3, -4] * isoRA[3, -3] +
                                           mpoB[idxMPO][-1, -2, 3, -4] * isoRB[3, -3]
        MPOC[idxMPO] = newTensor

        # bulk tensors
        for idxMPO in 2:(NA - 1)
            isoLA = isometry(space(mpoA[idxMPO], 1) ⊕ space(mpoB[idxMPO], 1),
                             space(mpoA[idxMPO], 1))
            isoLB = leftnull(isoLA)
            isoRA = isometry(space(mpoA[idxMPO], 3)' ⊕ space(mpoB[idxMPO], 3)',
                             space(mpoA[idxMPO], 3)')'
            isoRB = rightnull(isoRA)
            @tensor newTensor[-1 -2; -3 -4] := isoLA[-1, 1] * mpoA[idxMPO][1, -2, 3, -4] *
                                               isoRA[3, -3] +
                                               isoLB[-1, 1] * mpoB[idxMPO][1, -2, 3, -4] *
                                               isoRB[3, -3]
            MPOC[idxMPO] = newTensor
        end

        # right boundary tensor
        idxMPO = NA
        isoLA = isometry(space(mpoA[idxMPO], 1) ⊕ space(mpoB[idxMPO], 1),
                         space(mpoA[idxMPO], 1))
        isoLB = leftnull(isoLA)
        @tensor newTensor[-1 -2; -3 -4] := isoLA[-1 1] * mpoA[idxMPO][1 -2 -3 -4] +
                                           isoLB[-1 1] * mpoB[idxMPO][1 -2 -3 -4]
        MPOC[idxMPO] = newTensor
    end
    return SparseMPO(MPOC)
end
Base.:-(mpoA::SparseMPO, mpoB::SparseMPO) = mpoA + (-mpoB)

# multiplication
function Base.:*(M::SparseMPO, b::Number)
    newTensors = copy(M.mpoTensors)
    newTensors[1] *= b
    return SparseMPO(newTensors)
end
Base.:*(b::Number, M::SparseMPO) = M * b

function orthogonalizeMPO!(finiteMPO::SparseMPO, orthCenter::Int)
    """ Function to bring MPO into mixed canonical form with orthogonality center at site 'orthCenter' """

    # bring sites 1 to orthCenter - 1 into left-orthogonal form
    for siteIdx in 1:+1:(orthCenter - 1)
        (Q, R) = leftorth(finiteMPO[siteIdx], ((4, 1, 2), (3,)); alg = QRpos())
        finiteMPO[siteIdx + 0] = permute(Q, ((2, 3), (4, 1)))
        finiteMPO[siteIdx + 1] = permute(R *
                                         permute(finiteMPO[siteIdx + 1], ((1,), (2, 3, 4))),
                                         ((1, 2), (3, 4)))
    end

    # bring sites orthCenter + 1 to N into right-canonical form
    for siteIdx in length(finiteMPO):-1:(orthCenter + 1)
        (L, Q) = rightorth(finiteMPO[siteIdx], ((1,), (2, 3, 4)); alg = LQpos())
        finiteMPO[siteIdx - 1] = permute(permute(finiteMPO[siteIdx - 1],
                                                 ((4, 1, 2), (3,))) *
                                         L, ((2, 3), (4, 1)))
        finiteMPO[siteIdx - 0] = permute(Q, ((1, 2), (3, 4)))
    end
    return finiteMPO
end
function orthogonalizeMPO(finiteMPO::SparseMPO, args...)
    newMPO = copy(finiteMPO)
    orthogonalizeMPO!(newMPO, args...)
    return newMPO
end

function normMPO(finiteMPO::SparseMPO)
    """ Returns the norm of the MPO after bringing it into canonical form """

    orthogonalizeMPO!(finiteMPO, 1)
    normMPO = real(tr(finiteMPO[1]' * finiteMPO[1]))
    return normMPO
end

function normalizeMPO(finiteMPO::SparseMPO)
    """ Brings MPO into canonical form and returns normalized MPO """

    # bring finiteMPO into canonical form
    finiteMPO = orthogonalizeMPO(finiteMPO, 1)
    normMPO = real(tr(finiteMPO[1]' * finiteMPO[1]))
    finiteMPO[1] /= sqrt(normMPO)
    return finiteMPO
end

function applyMPO(finiteMPO::SparseMPO,
                  finiteMPS::SparseMPS;
                  truncErr::Float64 = 1e-6,
                  maxDim::Int64 = 2500,
                  compressionAlg::String = "variationalContraction",)
    """ Applies finiteMPO to finiteMPS and compressed the MPS to bond dimension 'maxDim' """
    """ Algorithm 'densityMatrix' is exact, algorithm 'zipUp' is faster but less accurate """

    # make copy of finiteMPS
    compressedMPS = copy(finiteMPS)

    # get length of finiteMPS
    N = length(compressedMPS)

    if compressionAlg == "densityMatrix"

        # construct MPO environments
        mpoEnvL = Vector{TensorMap{ComplexF64}}(undef, N)
        mpoEnvR = Vector{TensorMap{ComplexF64}}(undef, N)

        # initialize end-points of mpoEnvL and mpoEnvR
        mpoEnvL[1] = ones(ComplexF64,
                          space(compressedMPS[1], 1) ⊗ space(finiteMPO[1], 1),
                          space(finiteMPO[1], 1) ⊗ space(compressedMPS[1], 1))
        mpoEnvR[N] = ones(ComplexF64,
                          space(compressedMPS[N], 3)' ⊗ space(finiteMPO[N], 3)',
                          oneunit(spacetype(compressedMPS[N])))

        # compute mpoEnvL
        for siteIdx in 1:(N - 1)
            @tensor mpoEnvL[siteIdx + 1][-1 -2; -3 -4] := mpoEnvL[siteIdx][1, 2, 4, 6] *
                                                          conj(compressedMPS[siteIdx][1, 3,
                                                                                      -1]) *
                                                          conj(finiteMPO[siteIdx][2, 5, -2,
                                                                                  3]) *
                                                          finiteMPO[siteIdx][4, 5, -3, 7] *
                                                          compressedMPS[siteIdx][6, 7, -4]
        end

        # sweep from right to left and compress MPO * MPS
        for siteIdx in N:-1:1

            # construct density matrix
            @tensor rho[-1 -2; -3 -4] := mpoEnvL[siteIdx][1, 2, 4, 5] *
                                         conj(compressedMPS[siteIdx][1, 3, 9]) *
                                         conj(finiteMPO[siteIdx][2, -1, 10, 3]) *
                                         finiteMPO[siteIdx][4, -3, 8, 6] *
                                         compressedMPS[siteIdx][5, 6, 7] *
                                         conj(mpoEnvR[siteIdx][9, 10, -2]) *
                                         mpoEnvR[siteIdx][7, 8, -4]
            rho /= tr(rho)

            # make eigen decomposition of rho and truncate to maximal bond dimension
            U, S, V = tsvd(rho; trunc = truncdim(maxDim) & truncerr(truncErr))
            # U, S, V = tsvd(rho, trunc = truncerr(truncErr));

            # compute right environment for the site to the left
            if siteIdx > 1
                @tensor mpoEnvR[siteIdx - 1][-1 -2; -3] := compressedMPS[siteIdx][-1, 3,
                                                                                  1] *
                                                           finiteMPO[siteIdx][-2, 4, 2, 3] *
                                                           mpoEnvR[siteIdx][1, 2, 5] *
                                                           U[4, 5, -3]
            end

            # update MPS site
            compressedMPS[siteIdx] = permute(V, ((1, 2), (3,)))
        end

    elseif compressionAlg == "zipUp"

        # construct left and right isomorphism
        isomoL = isomorphism(fuse(space(compressedMPS[1], 1), space(finiteMPO[1], 1)),
                             space(compressedMPS[1], 1) ⊗ space(finiteMPO[1], 1))
        isomoR = isomorphism(space(compressedMPS[N], 3)' ⊗ space(finiteMPO[N], 3)',
                             fuse(space(compressedMPS[N], 3)' ⊗ space(finiteMPO[N], 3)'))

        # zip-up from left to right
        for siteIdx in 1:N
            if siteIdx < N
                @tensor localTensor[-1 -2; -3 -4] := isomoL[-1, 1, 3] *
                                                     compressedMPS[siteIdx][1, 2, -3] *
                                                     finiteMPO[siteIdx][3, -2, -4, 2]
                U, S, V = tsvd(localTensor,
                               ((1, 2),
                                (3, 4));
                               trunc = truncdim(maxDim) & truncerr(truncErr),
                               alg = TensorKit.SVD(),)
                # U, S, V = tsvd(localTensor, (1, 2), (3, 4), trunc = truncerr(truncErr), alg = TensorKit.SVD());
                compressedMPS[siteIdx] = permute(U, ((1, 2), (3,)))
                isomoL = S * V
            else
                @tensor compressedMPS[siteIdx][-1 -2; -3] := isomoL[-1, 1, 3] *
                                                             compressedMPS[siteIdx][1, 2,
                                                                                    4] *
                                                             finiteMPO[siteIdx][3, -2, 5,
                                                                                2] *
                                                             isomoR[4, 5, -3]
            end
        end

        # orthogonalize MPS
        orthogonalizeMPS!(compressedMPS, 1)

    elseif compressionAlg == "variationalContraction"

        # construct MPO environments
        mpoEnvL = Vector{TensorMap{ComplexF64}}(undef, N)
        mpoEnvR = Vector{TensorMap{ComplexF64}}(undef, N)

        # initialize end-points of mpoEnvL and mpoEnvR
        mpoEnvL[1] = ones(ComplexF64,
                          space(compressedMPS[1], 1),
                          space(finiteMPO[1], 1) ⊗ space(finiteMPS[1], 1))
        mpoEnvR[N] = ones(ComplexF64,
                          space(finiteMPS[N], 3)' ⊗ space(finiteMPO[N], 3)',
                          space(compressedMPS[N], 3)')

        # compute mpoEnvR, since the MPS is in right-canonical form
        for siteIdx in N:-1:2
            mpoEnvR[siteIdx - 1] = update_MPOEnvR(mpoEnvR[siteIdx],
                                                  finiteMPS[siteIdx],
                                                  finiteMPO[siteIdx],
                                                  compressedMPS[siteIdx])
        end

        # variational sweep
        variationalOverlap = 1.0
        runVariationalSweep = true
        while runVariationalSweep

            # sweep L ---> R
            for siteIdx in 1:+1:(N - 1)

                # construct two-site tensor
                @tensor newTheta[-1 -2; -3 -4] := mpoEnvL[siteIdx][-1, 3, 1] *
                                                  finiteMPS[siteIdx][1, 2, 4] *
                                                  finiteMPO[siteIdx][3, -2, 6, 2] *
                                                  finiteMPS[siteIdx + 1][4, 5, 7] *
                                                  finiteMPO[siteIdx + 1][6, -3, 8, 5] *
                                                  mpoEnvR[siteIdx + 1][7, 8, -4]

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, ((1, 2), (3, 4));
                                  trunc = truncdim(maxDim) & truncerr(truncErr))
                # U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncerr(truncErr));
                S /= norm(S)
                U = permute(U, ((1, 2), (3,)))
                V = permute(S * V, ((1, 2), (3,)))

                # assign updated tensors
                compressedMPS[siteIdx + 0] = U
                compressedMPS[siteIdx + 1] = V

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx],
                                                      finiteMPS[siteIdx],
                                                      finiteMPO[siteIdx],
                                                      compressedMPS[siteIdx])
            end

            # sweep L <--- R
            for siteIdx in (N - 1):-1:1

                # construct two-site tensor
                @tensor newTheta[-1 -2; -3 -4] := mpoEnvL[siteIdx][-1, 3, 1] *
                                                  finiteMPS[siteIdx][1, 2, 4] *
                                                  finiteMPO[siteIdx][3, -2, 6, 2] *
                                                  finiteMPS[siteIdx + 1][4, 5, 7] *
                                                  finiteMPO[siteIdx + 1][6, -3, 8, 5] *
                                                  mpoEnvR[siteIdx + 1][7, 8, -4]

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, ((1, 2), (3, 4));
                                  trunc = truncdim(maxDim) & truncerr(truncErr))
                # U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncerr(truncErr));
                S /= norm(S)
                U = permute(U * S, ((1, 2), (3,)))
                V = permute(V, ((1, 2), (3,)))

                # assign updated tensors
                compressedMPS[siteIdx + 0] = U
                compressedMPS[siteIdx + 1] = V

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1],
                                                      finiteMPO[siteIdx + 1],
                                                      compressedMPS[siteIdx + 1])
            end

            # normalize MPS after sweep
            mpsNorm = real(tr(compressedMPS[1]' * compressedMPS[1]))
            compressedMPS[1] /= sqrt(mpsNorm)

            # compute overlap ⟨ϕ|ψ⟩ and check convergence
            newVarOverlap = dotMPS(compressedMPS, finiteMPS)
            diffOverlap = abs(variationalOverlap - newVarOverlap)
            if diffOverlap < 1e-6
                runVariationalSweep = false
            end
            variationalOverlap = newVarOverlap
        end
    end
    return compressedMPS
end

#--------------------------------------------------------------
# SparseMPO save and load functions
#--------------------------------------------------------------

function save_to_file(fileName::String, sparseMPO::SparseMPO; dictKey::String = "sparseMPO")
    sparseMPO = convert.(Dict, sparseMPO)
    return JLD2.save(fileName, dictKey, sparseMPO)
end

function load_from_file(fileName::String, dictKey::String = "sparseMPO")
    sparseMPO = JLD2.load(fileName, dictKey)
    return sparseMPO = SparseMPO(convert.(TensorMap, sparseMPO))
end
