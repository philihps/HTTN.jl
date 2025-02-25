function auto_corr(A, tau::Int64)
    """
    Find the correlation length 

    ∑_{i=1}^{N - τ} (x_i - μ)(x_{i + τ} - μ) / ∑_{i=1}^{N} (x_i - μ)^2
    """
    N = length(A)
    mean_A = sum(A) / N
    res = N / (N - tau) * sum((A[1:(N - tau)] .- mean_A) .* (A[(1 + tau):N] .- mean_A)) /
          sum((A[1:N] .- mean_A) .^ 2)

    return res
end

function int_autocorr_time(autoCorrs; cutoff = 0.0005)
    """
    Computes the integrated autocorrelation time τ_int 
    from an array of autocorrelation values 

    Params:
    - cutoff: Threshold for selecting N_max where autoCorrs 
              is considered negligible (default: 0.05)
    """

    # Find N_max where autoCorrs effectively becomes negligible
    N_max = findfirst(k -> abs(autoCorrs[k]) < cutoff, 1:length(autoCorrs))

    if N_max === nothing  # If no clear cutoff is found, use full sum
        N_max = length(autoCorrs)
    end

    return sum(autoCorrs[1:N_max])
end

function compute_average(A)
    """
    Compute equilibrium average of A for correlated data
    <A>  = <A>_N ± σ_N        
    σ_N = √(<A^2>_N - <A>^2_N) / √(N / (2τ + 1))
    """
    N = length(A)
    mean_A = sum(A) / N
    taus = collect(1:(N ÷ 5))
    autoCorrs = [auto_corr(A, tau) for tau in taus]
    corrTime = int_autocorr_time(autoCorrs)
    println("Autocorrelation time is $(corrTime)")
    var = 1 / N * (sum(A .^ 2) / N - mean_A^2)
    if abs(var - 0) < 1e-8
        var = 0
    end
    stDev = sqrt(var)

    return mean_A, stDev * sqrt(1 + 2 * corrTime)
end
