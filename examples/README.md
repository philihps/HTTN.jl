# Guide for example files

## `metts_example.jl`
The many parameters of a many-body problem
- Display parameters
    - `modeOrdering`: ordering of momentum modes
        - `0 -> -2  -1   0  1  2`
        - `1 -> 0  -1  1  -2  2`

The many spaces of a many-body problem
- Space objects and parameters
    - `physSpaces`: physical spaces for MPS based on model
        - `k=0 -> Rep[U₁](0=>2*nMaxZM + 1)`: degeneracy of eigenvalue `0` is `2*nMaxZM + 1`
        - `k≠0 -> Rep[U₁](n_i*k=>1)`: degeneracy of eigenvalue `n_i*k` for `n_i = 0,...,nMax` is `1`
    - `virtSpaces`: virtual spaces for MPS such that for each local tensor, total momenta of outgoing legs = momenta of incoming leg
    - `removeDegeneracy`: option to remove degeneracy due to fusion of outgoing legs' spaces
