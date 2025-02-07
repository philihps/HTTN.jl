function removeDegeneracyQN(vecSpace; degenCutOff::Int64 = 1)
    """ Function to remove degeneracies of QNs """

    qnSectors = vecSpace.dims
    momentumQNs = [productSector.charge for productSector in keys(qnSectors)]
    momentumDGs = [degeneracyValue for degeneracyValue in values(qnSectors)]
    truncatedVectorSpace = Rep[U₁]([momentumQNs[qnIdx] => min(momentumDGs[qnIdx],
                                                              degenCutOff)
                                    for
                                    qnIdx in eachindex(momentumQNs)])
    return truncatedVectorSpace
end

# function constructPhysSpaces(momentumModes::Vector{Int64}, occPerSite::Vector{Int64})
#     """ Constructs vector spaces for physical bond indices of the MPS """

#     # get number of momentum modes
#     numSites = length(momentumModes);

#     # determine physSpaces
#     physSpaces = Vector{typeof(U1Space())}(undef, numSites);
#     for siteIdx = 1 : numSites
#         momentumVal = momentumModes[siteIdx];
#         maxOccupation = occPerSite[siteIdx];
#         if momentumVal == 0
#             physSpaces[siteIdx] = U1Space(0 => (2 * maxOccupation + 1));
#         else
#             physSpaces[siteIdx] = U1Space([momentumVal * occup => 1 for occup = 0 : +maxOccupation]);
#         end
#     end
#     return physSpaces;

# end

function infimum_larger_deg(V1::GradedSpace, V2::GradedSpace)
    """
    Construct the union of IRREPS of V1 and V2 with larger degeneracy
    """
    if V1.dual == V2.dual
        infimumSpace = typeof(V1)(c => min(dim(V1, c), dim(V2, c))
                                  for c in
                                      union(sectors(V1), sectors(V2)), dual in V1.dual)
        infimumSpace = [(c, max(dim(V1, c), dim(V2, c))) for c in sectors(infimumSpace)]
        infimumSpace = U1Space(infimumSpace)
        return infimumSpace
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function constructVirtSpaces(physSpaces::Vector{S}, qnL::S, qnR::S;
                             removeDegeneracy::Bool = true,
                             degenCutOff::Int64 = 1) where {S<:ElementarySpace}
    """ Constructs vector spaces for virtual bond indices of the MPS """

    # get number of momentum modes
    numSites = length(physSpaces)

    # create virtual vector spaces L
    virtSpaces_L = Vector{GradedSpace}()
    virtSpaces_L = vcat(virtSpaces_L, qnL)
    for ind in 1:+1:numSites
        spaceL = virtSpaces_L[end]
        spaceP = physSpaces[ind]
        spaceR = fuse(spaceL, spaceP)
        if removeDegeneracy
            spaceR = removeDegeneracyQN(spaceR; degenCutOff = degenCutOff)
        end
        virtSpaces_L = vcat(virtSpaces_L, [spaceR])
    end

    # create virtual vector spaces R
    virtSpaces_R = Vector{GradedSpace}()
    virtSpaces_R = vcat(qnR, virtSpaces_R)
    for ind in numSites:-1:1
        spaceR = virtSpaces_R[1]
        spaceP = physSpaces[ind]
        spaceL = fuse(conj(flip(spaceP)), spaceR)
        if removeDegeneracy
            spaceL = removeDegeneracyQN(spaceL)
        end
        virtSpaces_R = vcat([spaceL], virtSpaces_R)
    end

    # combine virtual vector spaces
    virtSpaces = [infimum(virtSpaces_L[siteIdx], virtSpaces_R[siteIdx])
                  for siteIdx in 1:(numSites + 1)]

    return virtSpaces
end

function generateKroneckerDeltaMPS(physSpaces::Vector{<:Union{ElementarySpace,
                                                              CompositeSpace{ElementarySpace}}};
                                   qnL::ElementarySpace = U1Space(0 => 1),
                                   qnR::ElementarySpace = U1Space(0 => 1),
                                   removeDegeneracy::Bool = true,
                                   degenCutOff::Int64 = 1,)
    """ Function to generate global momentum-conserving MPS to be contracted with local vertex operators to generate full interaction """

    # get number of momentum modes
    numSites = length(physSpaces)

    # create virtual vector spaces L
    virtSpaces_L = Vector{GradedSpace}()
    virtSpaces_L = vcat(virtSpaces_L, qnL)
    for ind in 1:+1:numSites
        spaceL = virtSpaces_L[end]
        spaceP = physSpaces[ind]
        spaceR = fuse(spaceL, spaceP)
        if removeDegeneracy
            spaceR = removeDegeneracyQN(spaceR; degenCutOff = degenCutOff)
        end
        virtSpaces_L = vcat(virtSpaces_L, [spaceR])
    end

    # create virtual vector spaces R
    virtSpaces_R = Vector{GradedSpace}()
    virtSpaces_R = vcat(qnR, virtSpaces_R)
    for ind in numSites:-1:1
        spaceR = virtSpaces_R[1]
        spaceP = physSpaces[ind]
        spaceL = fuse(conj(flip(spaceP)), spaceR)
        if removeDegeneracy
            spaceL = removeDegeneracyQN(spaceL)
        end
        virtSpaces_R = vcat([spaceL], virtSpaces_R)
    end

    # combine virtual vector spaces
    virtSpaces = [infimum(virtSpaces_L[siteIdx], virtSpaces_R[siteIdx])
                  for siteIdx in 1:(numSites + 1)]

    # construct MPS with physSpaces and virtSpaces
    deltaMPS = Vector{TensorMap{ComplexF64}}(undef, numSites)
    for siteIdx in 1:numSites
        deltaMPS[siteIdx] = ones(ComplexF64,
                                 virtSpaces[siteIdx] ⊗ physSpaces[siteIdx],
                                 virtSpaces[siteIdx + 1])
    end
    return SparseMPS(deltaMPS)
end
