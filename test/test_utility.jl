using HTTN
using Printf
using TensorKit
using Test

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

@testset "Test utility functions for vector spaces" begin
    V1 = U1Space(0 => 1, -2 => 1, -4 => 1, -6 => 1)
    V2 = U1Space((0 => 126, 1 => 84, -1 => 147, 2 => 42, -2 => 168, 3 => 21, -3 => 168,
                  -4 => 168, -5 => 147, -6 => 126, -7 => 84, -8 => 42, -9 => 21))
    infSpace = infimum_larger_deg(V1, V2)

    ans = [(0, 126), (-2, 168), (-4, 168), (-6, 126)]

    for (i, sec) in enumerate(sectors(infSpace))
        @test (sec.charge, dim(infSpace, sec)) == ans[i]
    end
end

nothing
