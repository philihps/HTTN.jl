using HTTN
using Printf
using TensorKit
using Test

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
