using HTTN
using Test

Ti = time()
include("test_basis_extension.jl")
include("test_BT_1_mode.jl")
include("test_BT_3_modes.jl")
include("test_METTS_aux.jl")
include("test_TDVP_DMRG_METTS.jl")
include("test_utility.jl")
include("test_sampling.jl")

Tf = time()
printstyled("Finished all tests in ",
            string(round((Tf - Ti) / 60; sigdigits = 3)),
            " minutes."; bold = true, color = Base.info_color())
println()

@testset "Run all tests" verbose = true begin
    using Aqua
    Aqua.test_all(HTTN; persistent_tasks = false)
end
