
"""
    function compress_MPO(vecTensor::Vector{TensorMap{ComplexF64}}; truncError = 1e-14)

    Function to compress an MPO with truncError the singular values
"""
function compress_MPO(vecTensor::SparseMPO; truncError::Float64 = 1e-14, verbose::Int64 = 0)
    matMPO = convert_MPO_33block(vecTensor)
    matMPO, Le, Re, Ss = compress_MPO(matMPO; truncError = truncError, verbose)
    return convert_33block_MPO(matMPO; leftEnvironment = Le, rightEnvironment = Re), Ss
end

"""
    function compress_MPO(vecTensor::Vector{Array{TensorMap}}; truncError = 1e-14)

    Example function for the optimization of the MPO.
    We assume a representation as a set of 3x3 matrix of tensors
"""
function compress_MPO(vecTensor::Vector{Array{TensorMap}}; truncError::Float64 = 1e-14,
                      verbose::Int64 = 0)

    # bring the MPO in a well-defined canonical form
    vecTensor, rightEnv = left_canonicalize(vecTensor; verbose)
    vecTensor, leftEnv = right_canonicalize(vecTensor; verbose)

    # perform actual truncation of the MPO
    vecTensor, rightEnv = left_canonicalize(vecTensor; rightEnvironment = rightEnv,
                                            truncation = true, truncError, verbose)
    vecTensor, leftEnv, Ss = right_canonicalize(vecTensor; leftEnvironment = leftEnv,
                                                truncation = true, truncError, verbose)

    # note it can be useful to repeat once or twice this truncation step for _very_ large MPO
    return vecTensor, leftEnv, rightEnv, Ss
end

"""
    function convert_MPO_33block(vecTensor::Vector{TensorMap{ComplexF64}})

    Converts a MPO represented by a vector of TensorMap into a matrix of MPO representations
    On all sites, we build a 3x3 matrix
    
    On site 1 it is of the form
    1  W  0
    0  0  0
    0  0  1
    On sites (2 : L-1) it is of the form
    1  0  0
    0  W  0
    0  0  1
    On site  L it is of the form
    1  0  0
    0  0  W
    0  0  1
    
    We have left and right boundaries conditions defined by
    left (1 0 0), right (0, 0, 1)
    The 0 tensors are such that lines and columns have well-defined indices

    Assumptions: we assume all TensorMap are organised in the following form:
    T[-1 -2; -3 -4] with 2 and 4 are the local physical dimensions
    1 is the left virtual space
    3 is the right virtual space

"""
function convert_MPO_33block(vecTensor::SparseMPO)
    L = length(vecTensor)
    mpo = Vector{Array{eltype(vecTensor)}}(undef, L)
    for siteIndex in eachindex(mpo)

        # initialize new 3x3 MPO block
        localVector = Array{eltype(vecTensor)}(undef, 3, 3)

        # get physical and virtual spaces
        physSpace = vecTensor[siteIndex].codom[2]
        virtSpaceL = vecTensor[siteIndex].codom[1]
        virtSpaceR = vecTensor[siteIndex].dom[1]

        # set local identity blocks
        localVector[1, 1] = TensorKit.id(U1Space(0 => 1) ⊗ physSpace)
        localVector[3, 3] = TensorKit.id(U1Space(0 => 1) ⊗ physSpace)

        # set local zero blocks
        localVector[1, 3] = zeros(ComplexF64, U1Space(0 => 1) ⊗ physSpace,
                                  U1Space(0 => 1) ⊗ physSpace)
        localVector[3, 1] = zeros(ComplexF64, U1Space(0 => 1) ⊗ physSpace,
                                  U1Space(0 => 1) ⊗ physSpace)
        for x in [1, 3]
            localVector[x, 2] = zeros(ComplexF64, U1Space(0 => 1) ⊗ physSpace,
                                      virtSpaceR ⊗ physSpace)
            localVector[2, x] = zeros(ComplexF64, virtSpaceL ⊗ physSpace,
                                      U1Space(0 => 1) ⊗ physSpace)
        end

        # set remaining blocks
        if siteIndex == 1
            localVector[1, 2] = vecTensor[siteIndex]
            localVector[2, 2] = zeros(Float64, virtSpaceL ⊗ physSpace,
                                      virtSpaceR ⊗ physSpace)
        elseif siteIndex == L
            localVector[2, 2] = zeros(Float64, virtSpaceL ⊗ physSpace,
                                      virtSpaceR ⊗ physSpace)
            localVector[2, 3] = vecTensor[siteIndex]
        else
            localVector[2, 2] = vecTensor[siteIndex]
        end
        mpo[siteIndex] = localVector
    end

    return mpo
end

"""
    function convert_33block_MPO(vecTensor::Vector{Array{TensorMap}}; leftEnvironment = undef, rightEnvironment = undef)

    Convert a MPO represented by a matrix of MPO representation to a vector of Tensor, see the function convert_MPO_33block
    We contract the leftEnvironment with vecTensor[1] (resp. rightEnvironment with vecTensor[end])
    If they are not defined, they take their default value (1, 0, ..., 0) and (0, ..., 0, 1).

    Assumptions: we assume all TensorMap are organised in the following form:
    T[-1 -2; -3 -4] with 2 and 4 are the local physical dimensions
    1 is the left virtual space
    3 is the right virtual space

"""
function convert_33block_MPO(vecTensor::Vector{Array{TensorMap}}; leftEnvironment = undef,
                             rightEnvironment = undef)
    L = length(vecTensor)
    mpoTensors = Vector{eltype(eltype(vecTensor))}(undef, L)
    for siteIndex in eachindex(mpoTensors)

        # get local 3x3 block
        localMatrix = vecTensor[siteIndex]

        # compute projectors
        left_projs = generate_projectors([localMatrix[idx, 1].codom[1]
                                          for idx in axes(localMatrix, 1)]...)
        right_projs = generate_projectors([localMatrix[1, idy].dom[1]
                                           for idy in axes(localMatrix, 2)]...)

        # combine 3x3 block into single mpo tensor
        combined_tensor = zeros(eltype(localMatrix[1, 1]),
                                left_projs[1].codom[1] ⊗ localMatrix[1, 1].codom[2],
                                right_projs[1].codom[1] ⊗ localMatrix[1, 1].dom[2])
        for idx in axes(localMatrix, 1), idy in axes(localMatrix, 2)
            @tensor temp[-1 -2; -3 -4] := left_projs[idx][-1, 1] *
                                          localMatrix[idx, idy][1, -2, 3, -4] *
                                          right_projs[idy]'[3, -3]
            combined_tensor += temp
        end
        mpoTensors[siteIndex] = combined_tensor
    end

    # initalize left environment
    if leftEnvironment == undef
        leftEnvironment = Vector{eltype(vecTensor[1])}(undef, size(vecTensor[1], 1))
        rightspace = vecTensor[1][1, 1].codom[1]
        leftEnvironment[1] = ones(eltype(vecTensor[1][2, 2]), U1Space(0 => 1),
                                  rightspace)
        for idx in 2:length(leftEnvironment)
            rightspace = vecTensor[1][idx, 1].codom[1]
            leftEnvironment[idx] = zeros(eltype(vecTensor[1][2, 2]),
                                         U1Space(0 => 1), rightspace)
        end
    end
    right_projs = generate_projectors([leftEnvironment[idx].dom[1]
                                       for idx in eachindex(leftEnvironment)]...)
    new_left = zeros(eltype(leftEnvironment[1]),
                     leftEnvironment[1].codom[1],
                     right_projs[1].codom[1])
    for idx in eachindex(leftEnvironment)
        @tensor temp[-1; -2] := leftEnvironment[idx][-1; 1] * right_projs[idx]'[1, -2]
        new_left += temp
    end
    localtensor = mpoTensors[1]
    @tensor temp[-1 -2; -3 -4] := new_left[-1, 1] * localtensor[1, -2, -3, -4]
    mpoTensors[1] = temp

    # initalize right environment
    if rightEnvironment == undef
        rightEnvironment = Vector{eltype(vecTensor[1])}(undef, size(vecTensor[end], 2))
        for idy in 1:(length(rightEnvironment) - 1)
            leftSpace = vecTensor[end][1, idy].dom[1]
            rightEnvironment[idy] = zeros(eltype(vecTensor[end][2, 2]),
                                          leftSpace, U1Space(0 => 1))
        end
        leftSpace = vecTensor[end][1, end].dom[1]
        rightEnvironment[end] = ones(eltype(vecTensor[end][2, 2]), leftSpace,
                                     U1Space(0 => 1))
    end
    left_projs = generate_projectors([rightEnvironment[x].codom[1]
                                      for x in eachindex(rightEnvironment)]...)
    new_right = zeros(eltype(rightEnvironment[1]),
                      left_projs[1].codom[1],
                      rightEnvironment[1].dom[1])
    for idy in eachindex(rightEnvironment)
        @tensor temp[-1; -2] := left_projs[idy][-1; 1] * rightEnvironment[idy][1; -2]
        new_right += temp
    end
    localtensor = mpoTensors[end]
    @tensor temp[-1 -2; -3 -4] := localtensor[-1, -2, 3, -4] * new_right[3, -3]
    mpoTensors[L] = temp
    return SparseMPO(mpoTensors)
end

"""
    function generate_projectors(V1::GradedSpace, V2::GradedSpace...)

    Given the GradedSpaces V1, V2... return the list of the projectors on ⊕(V1, V2...)
    Projections are done in order
    Useful for the block manipulations

"""
function generate_projectors(V1::GradedSpace, V2::GradedSpace...)
    Vjoined = ⊕(V1, V2...)
    proj1 = zeros(Float64, Vjoined, V1)
    shifts = Dict()
    for (c, b) in blocks(proj1)
        l = size(b, 2)
        shifts[c] = l
        for x in 1:l
            b[x, x] = 1
        end
    end
    projs = [proj1]
    for (j, V) in enumerate(V2)
        proj2 = zeros(Float64, Vjoined, V)
        for (c, b) in blocks(proj2)
            l = size(b, 2)
            shift = haskey(shifts, c) ? shifts[c] : 0
            for x in 1:l
                b[x + shift, x] = 1
            end
            if haskey(shifts, c)
                shifts[c] += l
            else
                shifts[c] = l
            end
        end
        append!(projs, [proj2])
    end
    return projs
end

"""
    function left_canonicalize(matrixMPO; rightEnvironment = undef, truncation::Bool = false, truncError = 1e-14, tol = 1e-12, verbose = 0)

    Given a vector of tensors in the matrixMPO representation (cf. convert_MPO_33block), return a left canonical form and the right boundary
    We follow the Parker et al https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147 algorithm.
    As a reminder, the key step is, moving from left to right
    1 a c    1 af c       1  t  0
    0 W b =  0 Wf b   x   0  R  0
    0 0 1    0 0  1       0  0  1
    where (1, af) = 0 for the operator scalar product)
           [af   is orthonormal on the right for the operator scalar product Wf]

    Optional arguments:
    - rightEnvironment : boundary term. If undef, it takes the default value (0, ..., 0, 1)
    - truncation::Bool : whether we perform a SVD truncation on the R intermediate matrix.
      We raise a warning if the t coefficient is non-zero within tol
    - truncError: the svd truncError
    - tol: warning for the truncation
    - verbose -> control the verbosity of the function
"""
function left_canonicalize(matrixMPO;
                           rightEnvironment = undef,
                           truncation::Bool = false,
                           truncError::Float64 = 1e-14,
                           tol::Float64 = 1e-12,
                           verbose::Int64 = 0,)
    matMPO = deepcopy(matrixMPO)
    L = length(matMPO)
    Ss = TensorMap[]

    # if undef, we introduce the right boundary tensor (0, 0, 1)
    if rightEnvironment == undef
        if verbose > 1
            println("Building the right environment")
        end
        rightEnvironment = Vector{eltype(matrixMPO[1])}(undef, 3)
        leftSpace = matMPO[end][1, 1].dom[1]
        rightEnvironment[1] = zeros(eltype(matMPO[end][2, 2]), leftSpace,
                                    U1Space(0 => 1))
        leftSpace = matMPO[end][1, 2].dom[1]
        rightEnvironment[2] = zeros(eltype(matMPO[end][2, 2]), leftSpace,
                                    U1Space(0 => 1))
        leftSpace = matMPO[end][1, 3].dom[1]
        rightEnvironment[3] = ones(eltype(matMPO[end][2, 2]), leftSpace,
                                   U1Space(0 => 1))
    else
        rightEnvironment = deepcopy(rightEnvironment)
    end

    # sweep from left --> right
    for siteIndex in 1:L
        if verbose > 1
            println("Canonicalizing site $siteIndex")
        end

        # input matrix representation
        # 1 a c
        # 0 W b
        # 0 0 1

        # first we orthogonalize the first two columns between themselves
        # 1 a c    1 at c     1  t  0
        # 0 W b =  0 W  b  x  0  1  0
        # 0 0 1    0 0  1     0  0  1
        # with at = a - 1 * (1, a)
        id = matMPO[siteIndex][1, 1]
        a = matMPO[siteIndex][1, 2]
        localSpace = id.codom[2]
        localDim = dim(localSpace)
        @tensor t[-1; -2] := (conj(id[1 2; -1 4]) * a[1 2; -2 4]) / localDim
        @tensor at[-1 -2; -3 -4] := id[-1 -2; 3 -4] * t[3; -3]
        at = a - at

        # now that we need to do a QR on the second column to orthogonalize the different vectors
        # 1 at c    1 af c     1  0  0
        # 0 W  b =  0 Wf b  x  0  R  0
        # 0 0  1    0 0  1     0  0  1
        # to do that, we need to tensorsum af and Wf on their left index. It is convenient to define projectors
        W = matMPO[siteIndex][2, 2]
        leftSpace_at = at.codom[1]
        leftSpace_W = W.codom[1]
        proj_at, proj_W = generate_projectors(leftSpace_at, leftSpace_W)
        @tensor fused_tensor[-1 -2; -3 -4] := proj_at[-1, 1] * at[1, -2, -3, -4] +
                                              proj_W[-1, 1] * W[1, -2, -3, -4]

        # # we then do a simple QR and reproject
        # Q, R = leftorth(fused_tensor, (1, 2, 4), (3, ), alg = QRpos());
        # Q = Q * sqrt(localDim); # proper normalization
        # R = R / sqrt(localDim);

        # check dimensions of tensors before the decomposition and use RQ or QR
        tensorDimensions = dim.([space(fused_tensor, idx) for idx in 1:4])
        if prod(tensorDimensions[1:3]) >= tensorDimensions[4]
            Q, R = leftorth(fused_tensor, ((1, 2, 4), (3,)); alg = QRpos())
        else
            Q, R = leftorth(fused_tensor, ((1, 2, 4), (3,)); alg = QLpos())
        end
        Q = Q * sqrt(localDim) # proper normalization
        R = R / sqrt(localDim)
        Q = permute(Q, ((1, 2), (4, 3)))

        if norm(R) <= 1e-12
            if verbose > 1
                println("Dirty fix, need to think about it a bit")
            end
            Q = Q * 0
        end

        if truncation && norm(R) > 1e-12
            norm(t) > tol && println("The form is not perfectly canonical")
            U, S, V, err = tsvd(R; trunc = truncerr(truncError))
            @tensor Q[-1 -2; -3 -4] := Q[-1, -2, 3, -4] * U[3, -3]
            R = S * V
            append!(Ss, [S])
            if verbose > 0
                println("Position $siteIndex : truncation error $err for norm $(norm(S))")
            end
        end

        # split again the fused_tensor into af and Wf
        @tensor af[-1 -2; -3 -4] := proj_at'[-1, 1] * Q[1, -2, -3, -4]
        @tensor Wf[-1 -2; -3 -4] := proj_W'[-1, 1] * Q[1, -2, -3, -4]

        # assign updated matrices to matMPO
        newRightSpace = Q.dom[1]
        matMPO[siteIndex][1, 2] = af
        matMPO[siteIndex][2, 2] = Wf
        matMPO[siteIndex][3, 2] = TensorMap(zeros,
                                            eltype(matMPO[siteIndex][3, 2]),
                                            U1Space(0 => 1) ⊗ localSpace,
                                            newRightSpace ⊗ localSpace)

        # at this point, we have obtained
        # 1 a c     1 af c     1  0  0     1  t  0     1 af c     1  t  0
        # 0 W b  =  0 Wf b  x  0  R  0  x  0  1  0  =  0 Wf b  x  0  R  0
        # 0 0 1     0 0  1     0  0  1     0  0  1     0 0  1     0  0  1

        # we need to transfer the right matrix to the next site
        # 1  t  0     1 a c     1   a + t W   c + t d
        # 0  R  0  x  0 W d  =  0     R W      R d
        # 0  0  1     0 0 1     0     0         1
        if siteIndex != L
            if verbose > 1
                println("Moving (R, t) to the right")
            end

            newSiteIndex = siteIndex + 1
            W = matMPO[newSiteIndex][2, 2]
            d = matMPO[newSiteIndex][2, 3]
            id = matMPO[newSiteIndex][3, 3]
            localSpace = id.codom[2]

            # compute a + t * W
            @tensor temp[-1 -2; -3 -4] := t[-1, 1] * W[1, -2, -3, -4]
            matMPO[newSiteIndex][1, 2] += temp

            # compute R * W
            @tensor temp[-1 -2; -3 -4] := R[-1, 1] * W[1, -2, -3, -4]
            matMPO[newSiteIndex][2, 2] = temp

            # compute c + t * d
            @tensor temp[-1 -2; -3 -4] := t[-1, 1] * d[1, -2, -3, -4]
            matMPO[newSiteIndex][1, 3] += temp

            # compute R * d
            @tensor temp[-1 -2; -3 -4] := R[-1, 1] * d[1, -2, -3, -4]
            matMPO[newSiteIndex][2, 3] = temp

            # a bit of bookeeping
            matMPO[newSiteIndex][2, 1] = TensorMap(zeros,
                                                   eltype(matMPO[newSiteIndex][2, 1]),
                                                   R.codom[1] ⊗ localSpace,
                                                   U1Space(0 => 1) ⊗ localSpace)

        else
            if verbose > 1
                println("Updating the right environment")
            end

            # for the last site, instead we update the boundary
            # 1  t  0     a     a + t b
            # 0  R  0  x  b  =    R b
            # 0  0  1     c        c
            b = rightEnvironment[2]

            # compute a + t * b
            @tensor temp[-1; -2] := t[-1, 1] * b[1, -2]
            rightEnvironment[1] += temp

            # compute R * b
            @tensor temp[-1; -2] := R[-1, 1] * b[1, -2]
            rightEnvironment[2] = temp
        end
    end
    return matMPO, rightEnvironment, Ss
end

"""
    function right_canonicalize(matrixMPO; leftEnvironment = undef, truncation::Bool = false, truncError = 1e-14, tol = 1e-12, verbose = 0)

    Given a vector of tensors in the matrixMPO representation (cf convert_MPO_33block), return a right canonical form and the left boundary
    We follow Parker et al https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147 algorithm.
    As a reminder, the key step is, moving from right to left
    1 at c    1 0 0     1  a  c
    0 W  b =  0 R t  x  0  Wf bf
    0 0  1    0 0 1     0  0  1
    where (1, bf) = 0 for the operator scalar product)
           [Wf bf]  is orthonormal on the left for the operator scalar product

    Optional arguments:
    - leftEnvironment : boundary term. If undef, it takes the default value (1, 0, ..., )
    - truncation::Bool : whether we perform a SVD truncation on the R intermediate matrix.
      We raise a warning if the t coefficient is non-zero within tol
    - truncError: the svd truncError
    - tol: warning for the truncation
    - verbose -> control the verbosity of the function
"""
function right_canonicalize(matrixMPO;
                            leftEnvironment = undef,
                            truncation = false,
                            truncError::Float64 = 1e-14,
                            tol::Float64 = 1e-12,
                            verbose::Int64 = 0,)
    matMPO = deepcopy(matrixMPO)
    L = length(matMPO)
    Ss = TensorMap[]

    # if undef, we introduce the left boundary tensor (1, 0, 0)
    if leftEnvironment == undef
        if verbose > 1
            println("Building the left environment")
        end
        leftEnvironment = Vector{eltype(matrixMPO[1])}(undef, 3)
        rightSpace = matMPO[1][1, 1].codom[1]
        leftEnvironment[1] = TensorMap(ones, eltype(matMPO[1][2, 2]), U1Space(0 => 1),
                                       rightSpace)
        rightSpace = matMPO[1][2, 1].codom[1]
        leftEnvironment[2] = zeros(eltype(matMPO[1][2, 2]), U1Space(0 => 1),
                                   rightSpace)
        rightSpace = matMPO[1][3, 1].codom[1]
        leftEnvironment[3] = zeros(eltype(matMPO[1][2, 2]), U1Space(0 => 1),
                                   rightSpace)
    else
        leftEnvironment = deepcopy(leftEnvironment)
    end

    # sweep from right --> left
    for siteIndex in reverse(1:L)
        if verbose > 1
            println("Canonicalizing site $siteIndex")
        end

        # input matrix representation
        # 1 a c
        # 0 W b
        # 0 0 1

        # first we orthogonalize the last two rows between themselves
        # 1 a c    1 0 0     1  a  0
        # 0 W b =  0 1 t  x  0  W  bt
        # 0 0 1    0 0 1     0  0  1
        # with bt = b - 1 * (1, b)
        b = matMPO[siteIndex][2, 3]
        id = matMPO[siteIndex][3, 3]
        localSpace = id.codom[2]
        localDim = dim(localSpace)
        @tensor t[-1; -2] := (b[-1, 2, 3, 4] * conj(id[-2, 2, 3, 4])) / localDim
        @tensor bt[-1 -2; -3 -4] := t[-1, 1] * id[1, -2, -3, -4]
        bt = b - bt

        # ow that we need to do a RQ on the second row to orthogonalize the different vectors
        # 1 at c     1 0 0     1  a  c
        # 0 W  b  =  0 R t  x  0  Wf bf
        # 0 0  1     0 0 1     0  0  1
        # to do that, we need to tensorsum bf and Wf on their right index. It is convenient to define projectors
        W = matMPO[siteIndex][2, 2]
        rightSpace_bt = bt.dom[1]
        rightSpace_W = W.dom[1]
        proj_W, proj_bt = generate_projectors(rightSpace_W, rightSpace_bt)
        @tensor fused_tensor[-1 -2; -3 -4] := W[-1, -2, 3, -4] * proj_W'[3, -3] +
                                              bt[-1, -2, 3, -4] * proj_bt'[3, -3]

        # # we then do a simple RQ and reproject
        # R, Q = rightorth(fused_tensor, (1, ), (2, 3, 4), alg = RQpos());
        # Q = Q * sqrt(localDim); # proper normalization
        # R = R / sqrt(localDim);

        # we then do a simple LQ and reproject
        R, Q = rightorth(fused_tensor, ((1,), (2, 3, 4)); alg = LQpos())
        Q = Q * sqrt(localDim) # proper normalization
        R = R / sqrt(localDim)

        # # check dimensions of tensors before the decomposition and use RQ or LQ
        # tensorDimensions = dim.([space(fused_tensor, idx) for idx = 1 : 4]);
        # display(tensorDimensions)
        # if tensorDimensions[1] <= prod(tensorDimensions[2 : 4])
        #     R, Q = rightorth(fused_tensor, (1, ), (2, 3, 4), alg = RQpos());
        # else
        #     R, Q = rightorth(fused_tensor, (1, ), (2, 3, 4), alg = LQpos());
        # end
        # Q = Q * sqrt(localDim); # proper normalization
        # R = R / sqrt(localDim);
        # Q = permute(Q, (1, 2), (3, 4));

        if norm(R) <= 1e-12
            if verbose > 1
                println("Dirty fix, need to think about it a bit")
            end
            Q = Q * 0
        end

        if truncation && norm(R) > 1e-12
            norm(t) > tol && println("The form is not perfectly canonical")
            U, S, V, err = tsvd(R; trunc = truncerr(truncError))
            R = U * S
            @tensor Q[-1 -2; -3 -4] := V[-1, 1] * Q[1, -2, -3, -4]
            append!(Ss, [S])
            if verbose > 0
                println("Position $siteIndex : truncation error $err for norm $(norm(S))")
            end
        end

        # split again the fused_tensor into bf and Wf
        @tensor Wf[-1 -2; -3 -4] := Q[-1, -2, 3, -4] * proj_W[3, -3]
        @tensor bf[-1 -2; -3 -4] := Q[-1, -2, 3, -4] * proj_bt[3, -3]

        # assign updated matrices to matMPO
        newLeftSpace = Q.codom[1]
        matMPO[siteIndex][2, 3] = bf
        matMPO[siteIndex][2, 2] = Wf
        matMPO[siteIndex][2, 1] = TensorMap(zeros,
                                            eltype(matMPO[siteIndex][2, 1]),
                                            newLeftSpace ⊗ localSpace,
                                            U1Space(0 => 1) ⊗ localSpace)

        # we need to transfer the left matrix to the next site
        # 1  a  c     1 0 0     1     a R     c + a t
        # 0  W  b  x  0 R t  =  0     W R     W t + b
        # 0  0  1     0 0 1     0     0          1
        if siteIndex != 1
            if verbose > 1
                println("Moving (R, t) to the left")
            end

            newSiteIndex = siteIndex - 1
            id = matMPO[newSiteIndex][1, 1]
            a = matMPO[newSiteIndex][1, 2]
            W = matMPO[newSiteIndex][2, 2]
            localSpace = id.codom[2]

            # compute c + a * t
            @tensor temp[-1 -2; -3 -4] := a[-1, -2, 3, -4] * t[3, -3]
            matMPO[newSiteIndex][1, 3] += temp

            # compute b + W * t
            @tensor temp[-1 -2; -3 -4] := W[-1, -2, 3, -4] * t[3, -3]
            matMPO[newSiteIndex][2, 3] += temp

            # compute a * R
            @tensor temp[-1 -2; -3 -4] := a[-1, -2, 3, -4] * R[3, -3]
            matMPO[newSiteIndex][1, 2] = temp

            # compute W * R
            @tensor temp[-1 -2; -3 -4] := W[-1, -2, 3, -4] * R[3, -3]
            matMPO[newSiteIndex][2, 2] = temp

            # a bit of bookeeping
            matMPO[newSiteIndex][3, 2] = TensorMap(zeros,
                                                   eltype(matMPO[newSiteIndex][3, 2]),
                                                   U1Space(0 => 1) ⊗ localSpace,
                                                   R.dom[1] ⊗ localSpace)

        else
            if verbose > 1
                println("Updating the right environment")
            end

            # for the last site, instead we update the boundary
            # a     1  0  0       a
            # b  x  0  R  t  =   b R
            # c     0  0  1     c + bt
            b = leftEnvironment[2]

            # compute c + b * t
            @tensor temp[-1; -2] := b[-1, 1] * t[1, -2]
            leftEnvironment[3] += temp

            # compute b * R
            @tensor temp[-1; -2] := b[-1, 1] * R[1, -2]
            leftEnvironment[2] = temp
        end
    end
    return matMPO, leftEnvironment, Ss
end
