import numpy as np
import pdb

def get_online_quantile(scores, q_1, etas, alpha):
    """
    Computes the online quantile of a set of scores.
    :param scores: (np.array) The scores.
    :param q_1: (float) The quantile to compute.
    :param eta: (np.array) The sequence of learning rates.
    :return: (float) The sequence of online quantiles.
    """
    T = scores.shape[0]
    q = np.zeros(T)
    q[0] = q_1
    for t in range(T):
        err_t = (scores[t] > q[t]).astype(int)
        if t < T - 1:
            q[t + 1] = q[t] - etas[t] * (alpha - err_t)
    return q


def smooth_array(arr, window_size):
    # Create a window of ones of length window_size
    window = np.ones(window_size) / window_size
    
    # Use convolve to apply the window to the array
    # 'valid' mode returns output only where the window fits completely
    smoothed = np.convolve(arr, window, mode='same')
    
    return smoothed

def decay_ACI(Y, Y_hat, alpha, t_init=500,  max_band = 10):
    T = len(Y)

    scores = np.abs(Y - Y_hat)
    # Problem setup
    scores_val = scores[::2]
    scores = scores[1::2]
    T = len(scores) # How long to run for
    N_val = len(scores_val)
    etas_fixed = np.ones(T)*0.05
    epsilon = 0.1
    etas_decaying = np.array([1/(t**(1/2+epsilon)) for t in range(1, T+1)])
    q_1 = scores[0]
    q_star = np.quantile(np.concatenate([scores, scores_val]), 1-alpha)
   
    # Get the quantiles
    q_fixed = get_online_quantile(scores, q_1, etas_fixed, alpha)
    q_decaying = get_online_quantile(scores, q_1, etas_decaying, alpha)

    # Check coverage
    observed_coverages_fixed = (scores <= q_fixed).astype(int)
    observed_coverages_decaying = (scores <= q_decaying).astype(int)
    observed_coverages_oracle = (scores <= q_star).astype(int)

    # # Long-run coverage
    # LRC_fixed = np.cumsum(observed_coverages_fixed)/(np.arange(len(observed_coverages_fixed))+1)
    # LRC_decaying = np.cumsum(observed_coverages_decaying)/(np.arange(len(observed_coverages_decaying))+1)
    # LRC_qstar = np.cumsum(observed_coverages_oracle)/(np.arange(len(observed_coverages_oracle))+1)

    # # Rolling coverage
    # W = 1000
    # rolling_coverage_fixed = smooth_array(observed_coverages_fixed, W)
    # rolling_coverage_decaying = smooth_array(observed_coverages_decaying, W)
    # rolling_coverage_oracle = smooth_array(observed_coverages_oracle, W)

    # # Time-conditional coverage
    # time_coverage_fixed = (scores_val[:,None] <= q_fixed[None,:]).mean(axis=0)
    # time_coverage_decaying = (scores_val[:,None] <= q_decaying[None,:]).mean(axis=0)
    # time_coverage_qstar = (scores_val[:,None] <= q_star*np.ones_like(q_decaying)[None,:]).mean(axis=0)

    #return alpha_trajectory, adapt_err_seq, no_adapt_error_seq, (band_native, band_adapt)
    return q_fixed, q_decaying, (observed_coverages_fixed, observed_coverages_decaying, observed_coverages_oracle)
