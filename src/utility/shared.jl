function diagTM(T::AbstractTensorMap)
    """ Overloading of LinearAlgebra function diag for TensorMap type """

    diagElements = Vector{eltype(T)}()
    blockSectors = blocksectors(T)
    for blockIdx in blockSectors
        append!(diagElements, real.(diag(block(T, blockIdx))))
    end
    return sort(diagElements; rev = true)
end
