
import torch
import os
import numpy as np
from tqdm import tqdm
import math


def conformal_prediction_cptc(gt, preds, z_prob, z_mean, T=100, alpha=0.1, gamma=0.2, max_width=3,  min_residuals=25):
    """
    Conformal Prediction for Time series with Change Points (CPTC)
    
    Parameters:
    
    Returns:
    Prediction intervals for future time steps t > T
    """

    def A(gt, pred):
        return np.linalg.norm(gt - pred)
        

    all_coverages = []
    all_widths = []
    
    # Initialization of variables
    states = list(range(z_prob.shape[-1]))  # Get the list of states z

    prediction_intervals = {}  # Store prediction intervals Γ(xt) for t > T
    S_z = {z: [max_width] for z in states}  # Residuals for each state z
    alpha_z_T = {z: alpha for z in states}  # Initialize alpha_z,T for each state

    # Calculate prediction residuals for warm start data
    for t in range(T):
        #z_t = np.argmax(z_prob[t])
        for z, p_z_t in enumerate(z_prob[t]):
            if p_z_t > 0.3:
                S_z[z].append( A(gt[t], preds[t]) )

    # Add all residuals to "all" key, fallback for when state calibaration data is too little
    S_z["all"] = [A(gt[t], preds[t]) for t in range(T)] + [max_width]

    # Main loop for t > T
    for t in range(T, gt.shape[0]):
        # Initialize an empty prediction set for this time step
        prediction_intervals[t] = []

        # Loop over each state z
        for z in states:
            # Step 4: Calculate state probability p_z,t
            p_z_t = z_prob[t, z]
            if p_z_t == 0:
                continue

            # Step 5: Obtain state-specific confidence level alpha_hat_z,t
            alpha_hat_z_t = alpha_z_T[z]

            # Step 6: Calculate conformal prediction set Γ_z,t
            residuals = S_z[z]

            if len(residuals) < min_residuals:
                residuals = S_z["all"]

            # if residuals is empty, use all residuals
            if len(residuals) == 0:
                all_residuals = np.concatenate(list(S_z.values()))
                score = np.quantile(all_residuals, np.clip(1 - alpha_hat_z_t, 0, 1))
            else:
                score = np.quantile(residuals, np.clip(1 - alpha_hat_z_t, 0, 1))
            
            # mean, radius, probability of interval
            interval = (z_mean[t, z].squeeze(), score, z_prob[t, z])
            prediction_intervals[t].append(interval)

        # Step 7: Output confidence region Γ_t(x_t), here is union of all intervals   
        prediction_intervals[t] = sorted(prediction_intervals[t], key=lambda x: x[2])
        smallest_set = []
        running_sum = 0
        
        for interval in prediction_intervals[t]:
            if running_sum < 1-alpha:
                smallest_set.append(interval)
                running_sum += interval[2]
            else:
                break

        final_confidence_region = smallest_set

        #final_confidence_region = np.max(prediction_intervals[t])
        def get_width(intervals):
            width = 0
            for interval in intervals:
                mean, radius, _ = interval
                if len(mean.shape) <= 1:
                    return radius
                else:
                    # multivariate case. For this paper only 2d is supported
                    return math.pi * (radius ** 2)

        all_widths.append(get_width(final_confidence_region))

        # Step 8: Sample state at time t from predicted distribution
        z_hat_t = np.random.choice(states, p=z_prob[t])

        # Step 9: Observe true y_t (this will be given or simulated)
        y_t = gt[t]

        # Step 10: Calculate error
        def is_in_interval(final_confidence_region, y_t):
            for interval in final_confidence_region:
                mean, radius, _ = interval
                if len(y_t.shape) == 0 or y_t.shape[-1] == 1:
                    if y_t >= mean - radius and y_t <= mean + radius:
                        return True
                else:
                    if np.linalg.norm(y_t[:2] - mean[:2]) <= radius:
                        return True
            return False

        covered = is_in_interval(final_confidence_region, y_t)
        all_coverages.append(covered)
        err_t = int(not covered)

        # Step 11: Update alpha for the predicted state
        alpha_z_T[z_hat_t] = alpha_z_T[z_hat_t] + gamma * (alpha - err_t)

        # Step 12: Update nonconformity scores with residuals
        S_z[z_hat_t].append(A(preds[t], y_t))

    return all_coverages, all_widths, prediction_intervals
