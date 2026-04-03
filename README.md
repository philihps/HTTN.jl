# HTTN

[![Build Status](https://github.com/philihps/HTTN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/philihps/HTTN.jl/actions/workflows/CI.yml?query=branch%3Amain)

Hamiltonian Truncation Tensor Network (HTTN) method for the simulation of (1+1)d QFTs.

The package can be installed by running
```
pkg> add https://github.com/philihps/HTTN.jl
```

# Content of the package
HTTN.jl implements matrix product state (MPS) methods to simulate (1+1)d bosonic QFTs, with current support for the sine-Gordon model (sG) and the massive Schwinger model (mS). The MPS is set up in the basis of the free part of the Hamiltonian as a tensor product of decoupled harmonic oscillators. The interacting part is implemented as the sum of two vertex operator, generating long-range interaction between the modes. The length of the system corresponds to a UV cutoff in momentum space, while each bosonic Hilbert space is truncated to a finite number of occupations.
