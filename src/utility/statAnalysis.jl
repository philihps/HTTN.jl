"""
Julia implementation of https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
"""

using FFTW, LinearAlgebra

function nextPowTwo(n::Int)
    i = 1
    while i < n
        i <<= 1
    end
    return i
end

function findWindow(taus::Vector{<:Real}, c::Real)
    """
    Automated windowing procedure following Sokal (1989)
    """
    m = collect(1:length(taus)) .< c .* taus
    if any(m)
        return argmin(m)
    end
    return length(taus)
end

function computeAutoCorr1D(x::Vector{<:Real}, norm::Bool = true)
    """
    Compute autocorrelation function for 1D data
    """
    if length(size(x)) != 1
        throw(ArgumentError("invalid dimensions for 1D autocorrelation function"))
    end

    n = nextPowTwo(length(x))
    # Compute the FFT and then (from that) the auto-correlation function
    x_mean = sum(x) / length(x)

    f = fft(vcat(x .- x_mean, zeros(2 * n - length(x))))
    acf = real(ifft(f .* conj(f)))[1:length(x)]
    acf ./= 4 * n

    # Optionally normalize
    if norm
        acf ./= acf[1]
    end

    return acf
end

function computeAutoCorrTime(data, c::Real = 5.0)
    """
    Compute integrated autocorrelation time for parallel runs
    """
    f = zeros(size(data, 2))
    for i in 1:size(data, 1)
        f .+= computeAutoCorr1D(data[i, :])
    end
    f ./= size(data, 1)
    taus = 2.0 .* cumsum(f) .- 1.0
    window = findWindow(taus, c)
    return taus[window]
end

function computeAverage(A)
    """
    Compute equilibrium average of A for correlated data
    <A>  = <A>_N ± σ_N        
    σ_N = √(<A^2>_N - <A>^2_N) / √(N / (2τ + 1))
    """
    N = length(A)
    mean_A = sum(A) / N
    intCorrTime = computeAutoCorrTime(A)
    println("Autocorrelation time is $(intCorrTime)")
    var = 1 / N * (sum(A .^ 2) / N - mean_A^2)
    if abs(var - 0) < 1.0e-8
        var = 0
    end
    stDev = sqrt(var)

    return mean_A, stDev * sqrt(intCorrTime)
end
