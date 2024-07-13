#!/usr/bin/env julia

#------------------------------------------------------
# MPS types
#------------------------------------------------------

abstract type AbstractMPS end
abstract type AbstractFiniteMPS <: AbstractMPS end

struct SparseMPS{A<:AbstractTensorMap} <: AbstractFiniteMPS
    
    mpsTensors::Vector{A}

    function SparseMPS{A}(mpsTensors::Vector{A}) where {A<:AbstractTensorMap}
        return new{A}(mpsTensors)
    end

    function SparseMPS(mpsTensors::Vector{A}; normalizeMPS = false) where {A<:AbstractTensorMap}

        # bring MPS into right canonical form
        for siteIdx = length(mpsTensors) : -1 : 2
            (L, Q) = rightorth(mpsTensors[siteIdx], (1, ), (2, 3), alg = LQpos());
            normalizeMPS && normalize!(L)
            mpsTensors[siteIdx - 1] = permute(permute(mpsTensors[siteIdx - 1], (1, 2), (3, )) * L, (1, 2), (3, ));
            mpsTensors[siteIdx - 0] = permute(Q, (1, 2), (3, ));
        end

        return new{A}(mpsTensors)

    end

end


"""
getVirtualSpaceL(ψ::AbstractMPS, siteIdx::Int)
    
Returns the left virtual space of the MPS tensor at site 'siteIdx'.
This is equivalent to the right virtual space of the MPS tensor at site 'siteIdx - 1'.

"""
# function getVirtualSpaceL end

function getVirtualSpaceL(ψ::SparseMPS, siteIdx::Integer)
    return space(ψ.mpsTensors[siteIdx], 1);
end

"""
getVirtualSpaceR(ψ::AbstractMPS, siteIdx::Int)
    
Returns the right virtual space of the MPS tensor at site 'siteIdx'.
This is equivalent to the left virtual space of the MPS tensor at site 'siteIdx + 1'.

"""
# function getVirtualSpaceR end

function getVirtualSpaceR(ψ::SparseMPS, siteIdx::Integer)
    return dual(space(ψ.mpsTensors[siteIdx], 3));
end

"""
getPhysicalSpace(ψ::AbstractMPS, siteIdx::Int)
    
Returns the physical space of the MPS tensor at site 'siteIdx'.

"""
# function getPhysicalSpace end

function getPhysicalSpace(ψ::SparseMPS, siteIdx::Integer)
    return space(ψ.mpsTensors[siteIdx], 2);
end


#--------------------------------------------------------------
# SparseMPS constructors
#--------------------------------------------------------------

function SparseMPS(initMethod, elementType::DataType, physSpaces::Vector{<:Union{S, CompositeSpace{S}}}, virtSpaces::Vector{S}; normalizeMPS = true) where {S<:ElementarySpace}
    
    N = length(physSpaces);
    length(virtSpaces) == N + 1 || throw(DimensionMismatch("length of physical spaces ($N) and virtual spaces $(length(virtSpaces)) should differ by 1"))

    # construct MPS
    mpsTensors = [TensorMap(initMethod, elementType, virtSpaces[siteIdx] ⊗ physSpaces[siteIdx], virtSpaces[siteIdx + 1]) for siteIdx = 1 : N];
    return SparseMPS(mpsTensors, normalizeMPS = normalizeMPS)

end

function SparseMPS(physSpaces::Vector{<:Union{S, CompositeSpace{S}}}, virtSpaces::Vector{S}, inputState::SparseMPS; normalizeMPS = true) where {S<:ElementarySpace}
    
    # construct MPS
    numSites = length(inputState);
    chainCenter = Int((numSites + 1) / 2);
    mpsTensors = Vector{TensorMap}(undef, numSites);
    for siteIdx = 1 : numSites
        initTensor = 1e-0 * convert(Array, inputState[siteIdx]);
        siteTensor = TensorMap(randn, Float64, virtSpaces[siteIdx] ⊗ physSpaces[siteIdx], virtSpaces[siteIdx + 1]);
        if siteIdx == 1
            siteTensor = 1e-2 * convert(Array, siteTensor);
        else
            siteTensor = 1e-1 * convert(Array, siteTensor);
        end
        dimsInitTensor = size(initTensor);
        dimsSiteTensor = size(siteTensor);
        minTensorDim = min(dimsInitTensor, dimsSiteTensor);
        if siteIdx != chainCenter
            siteTensor[1 : minTensorDim[1], 1 : minTensorDim[2], 1 : minTensorDim[3]] = initTensor[1 : minTensorDim[1], 1 : minTensorDim[2], 1 : minTensorDim[3]];
        else
            siteTensor[1 : minTensorDim[1], 1 : minTensorDim[2], 1 : minTensorDim[3]] = initTensor[1 : minTensorDim[1], 1 : minTensorDim[2], 1 : minTensorDim[3]];
        end
        mpsTensors[siteIdx] = TensorMap(siteTensor, virtSpaces[siteIdx] ⊗ physSpaces[siteIdx], virtSpaces[siteIdx + 1]);
    end

    # construct MPS
    return SparseMPS(mpsTensors, normalizeMPS = normalizeMPS)

end


#--------------------------------------------------------------
# SparseMPS utilities
#--------------------------------------------------------------

Base.getindex(ψ::SparseMPS, idx) = ψ.mpsTensors[idx];
Base.setindex!(ψ::SparseMPS, mpsTensor, idx::Int) = (ψ.mpsTensors[idx] = mpsTensor)
Base.size(ψ::SparseMPS, args...) = size(ψ.mpsTensors, args...)
Base.length(ψ::SparseMPS) = length(ψ.mpsTensors)
Base.eltype(ψ::SparseMPS) = eltype(ψ.mpsTensors)
Base.iterate(ψ::SparseMPS, args...) = iterate(ψ.mpsTensors, args...)
Base.eachindex(ψ::SparseMPS, args...) = eachindex(ψ.mpsTensors, args...)
Base.copy(ψ::SparseMPS) = SparseMPS(copy(ψ.mpsTensors))
Base.lastindex(ψ::SparseMPS) = lastindex(ψ.mpsTensors)
function Base.similar(ψ::SparseMPS{A}) where {A}
    return SparseMPS{A}(similar(ψ.mpsTensors))
end

getLinkDimsMPS(ψ::SparseMPS) = dim.(vcat([getVirtualSpaceL(ψ, idx) for idx = 1 : length(ψ)], getVirtualSpaceR(ψ, length(ψ))))
maxLinkDimsMPS(ψ::SparseMPS) = maximum(getLinkDimsMPS(ψ))


# multiplication
function Base.:*(ψ::SparseMPS, b::Number)
    
    newTensors = copy(ψ.mpsTensors);
    newTensors[1] *= b;
    return SparseMPS(newTensors)
end
Base.:*(b::Number, ψ::SparseMPS) = ψ * b

function orthogonalizeMPS!(finiteMPS::SparseMPS, orthCenter::Int = 1)
    """ Function to bring MPS into mixed canonical form with orthogonality center at site 'orthCenter' """

    # bring sites 1 to orthCenter - 1 into left-orthogonal form
    for siteIdx = 1 : +1 : (orthCenter - 1)
        (Q, R) = leftorth(finiteMPS[siteIdx], (1, 2), (3, ), alg = QRpos());
        finiteMPS[siteIdx + 0] = permute(Q, (1, 2), (3, ));
        finiteMPS[siteIdx + 1] = permute(R * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, ));
    end

    # bring sites orthCenter + 1 to N into right-canonical form
    for siteIdx = length(finiteMPS) : -1 : (orthCenter + 1)
        (L, Q) = rightorth(finiteMPS[siteIdx], (1, ), (2, 3), alg = LQpos());
        finiteMPS[siteIdx - 1] = permute(permute(finiteMPS[siteIdx - 1], (1, 2), (3, )) * L, (1, 2), (3, ));
        finiteMPS[siteIdx - 0] = permute(Q, (1, 2), (3, ));
    end
    return finiteMPS;

end
function orthogonalizeMPS(finiteMPS::SparseMPS, args...)
    newMPS = copy(finiteMPS);
    orthogonalizeMPS!(newMPS, args...);
    return newMPS
end

function normMPS(finiteMPS::SparseMPS)
    """ Returns the norm of the MPS after bringing it into canonical form """

    orthogonalizeMPS!(finiteMPS, 1);
    normMPS = real(tr(finiteMPS[1]' * finiteMPS[1]));
    return normMPS;
end

function normalizeMPS(finiteMPS::SparseMPS)
    """ Brings MPS into canonical form and returns normalized MPS """

    # bring finiteMPS into canonical form
    orthogonalizeMPS!(finiteMPS, 1);
    normMPS = real(tr(finiteMPS[1]' * finiteMPS[1]));
    finiteMPS[1] /= sqrt(normMPS);
    return finiteMPS;

end

function dotMPS(mpsA::SparseMPS, mpsB::SparseMPS)
    """ Compute the overlap < mpsA | mpsB > """

    # get length of MPSs
    N = length(mpsA);
    N != length(mpsB) && throw(DimensionMismatch("lengths of MPS A ($N) and MPS B ($length(mpsB)) do not match"))

    # initialize overlap
    overlapMPS = TensorMap(ones, space(mpsA[1], 1), space(mpsB[1], 1));
    for siteIdx = 1 : N
        @tensor overlapMPS[-1; -2] := overlapMPS[1, 2] * conj(mpsA[siteIdx][1, 3, -1]) * mpsB[siteIdx][2, 3, -2];
    end
    overlapMPS = tr(overlapMPS);
    return overlapMPS;

end

# addition
function Base.:+(mpsA::SparseMPS, mpsB::SparseMPS)
    
    # get length of MPSs
    NA = length(mpsA);
    NB = length(mpsB);
    NA != NB && throw(DimensionMismatch("lengths of MPS A ($NA) and MPS B ($NB) do not match"))

    # add MPSs from left to right
    MPSC = Vector{TensorMap}(undef, NA);
    if NA == 1

        # combine single tensor
        idxMPS = 1;
        MPSC[idxMPS] = mpsA[idxMPS] + mpsB[idxMPS];

    else

         # left boundary tensor
        idxMPS = 1;
        isoRA = isometry(space(mpsA[idxMPS], 3)' ⊕ space(mpsB[idxMPS], 3)', space(mpsA[idxMPS], 3)')';
        isoRB = rightnull(isoRA);
        @tensor newTensor[-1 -2; -3] := mpsA[idxMPS][-1, -2, 3] * isoRA[3, -3] + mpsB[idxMPS][-1, -2, 3,] * isoRB[3, -3];
        MPSC[idxMPS] = newTensor;

        # bulk tensors
        for idxMPS = 2 : (NA - 1)
            isoLA = isometry(space(mpsA[idxMPS], 1) ⊕ space(mpsB[idxMPS], 1), space(mpsA[idxMPS], 1));
            isoLB = leftnull(isoLA);
            isoRA = isometry(space(mpsA[idxMPS], 3)' ⊕ space(mpsB[idxMPS], 3)', space(mpsA[idxMPS], 3)')';
            isoRB = rightnull(isoRA);
            @tensor newTensor[-1 -2; -3] := isoLA[-1, 1] * mpsA[idxMPS][1, -2, 3] * isoRA[3, -3] + isoLB[-1, 1] * mpsB[idxMPS][1, -2, 3] * isoRB[3, -3];
            MPSC[idxMPS] = newTensor;
        end

        # right boundary tensor
        idxMPS = NA;
        isoLA = isometry(space(mpsA[idxMPS], 1) ⊕ space(mpsB[idxMPS], 1), space(mpsA[idxMPS], 1));
        isoLB = leftnull(isoLA);
        @tensor newTensor[-1 -2; -3] := isoLA[-1, 1] * mpsA[idxMPS][1, -2, -3] + isoLB[-1, 1] * mpsB[idxMPS][1, -2, -3];
        MPSC[idxMPS] = newTensor;

    end
    return SparseMPS(MPSC);

end
Base.:-(mpsA::SparseMPS, mpsB::SparseMPS) = mpsA + (-1 * mpsB)


#--------------------------------------------------------------
# SparseMPS save and load functions
#--------------------------------------------------------------

function save_to_file(fileName::String, sparseMPS::SparseMPS; dictKey::String = "sparseMPS")
    sparseMPS = convert.(Dict, sparseMPS)
    JLD.save(fileName, dictKey, sparseMPS)
end

# function load_from_file(fileName::String, dictKey::String = "sparseMPS")
#     sparseMPS = JLD.load(fileName, dictKey)
#     sparseMPS = SparseMPS(convert.(TensorMap, sparseMPS))
# end