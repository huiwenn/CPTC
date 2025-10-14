### This file contains two functions for running adaptive conformal inference in order to reproduce Figures 1, 2, 4, 5, 6, and 7 in https://arxiv.org/abs/2106.00170. 

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
#from arch import arch_model

from scipy.optimize import minimize

class MultidimensionalQuantileRegression:
    def __init__(self, q=0.5, max_iter=1000):
        """
        Initialize the MultidimensionalQuantileRegression object.
        
        Parameters:
        q : float, optional (default=0.5)
            The quantile to estimate (0 < q < 1).
        max_iter : int, optional (default=1000)
            Maximum number of iterations for the optimization.
        """
        self.q = q
        self.max_iter = max_iter
        self.beta = None

    def fit(self, X, Y):
        """
        Fit the quantile regression model.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Y : array-like of shape (n_samples, n_targets)
            The target values.
        
        Returns:
        self : object
            Returns self.
        """
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]
        
        def objective(beta):
            beta = beta.reshape(n_features, n_targets)
            residuals = Y - X @ beta
            return np.sum(np.maximum(self.q * residuals, (self.q - 1) * residuals))
        
        beta_init = np.zeros(n_features * n_targets)
        
        result = minimize(objective, beta_init, method='BFGS', options={'maxiter': self.max_iter})
        
        self.beta = result.x.reshape(n_features, n_targets)
        
        return self

    def predict(self, X):
        """
        Predict using the fitted quantile regression model.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns:
        Y_pred : array of shape (n_samples, n_targets)
            The predicted values.
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using 'predict'.")
        
        return X @ self.beta

### Main method for forming election night predictions of county vote totals as in Figure 2

def ACI(Y, X, alpha, gamma, t_init=500, split_size=0.75, update_method="Simple", momentum_bw=0.95, max_band = 10):
    T = len(Y)
    # Initialize data storage variables
    alpha_trajectory = [alpha] * (T - t_init)
    adapt_err_seq = [0] * (T - t_init)
    no_adapt_error_seq = [0] * (T - t_init)
    alpha_t = alpha
    band_native = []
    band_adapt = []
    
    for t in range(t_init, T):
        # Split data into training and calibration set
        train_points = np.random.choice(t, size=int(split_size*t), replace=False)
        cal_points = np.setdiff1d(np.arange(t), train_points)
        X_train, Y_train = X[train_points], Y[train_points]
        X_cal, Y_cal = X[cal_points], Y[cal_points]
        
        # flatten multi-dimensional inputs
        # X_train = X_train.reshape(X_train.shape[0], -1)
        # X_cal = X_cal.reshape(X_cal.shape[0], -1)
        # Y_train = Y_train.reshape(Y_train.shape[0], -1)
        # Y_cal = Y_cal.reshape(Y_cal.shape[0], -1)
        #print(X_train.shape, X_cal.shape, Y_train.shape, Y_cal.shape)

        # Fit quantile regression on training setting
        if len(X_train.shape) > 1:
            model_upper = MultidimensionalQuantileRegression(q=1-alpha/2)
            model_lower = MultidimensionalQuantileRegression(q=alpha/2)
            res_upper = model_upper.fit(X_train, Y_train)
            res_lower = model_lower.fit(X_train, Y_train)
        else:
            model_upper = QuantReg(Y_train, X_train)
            model_lower = QuantReg(Y_train, X_train)
            res_upper = model_upper.fit(q=1-alpha/2)
            res_lower = model_lower.fit(q=alpha/2)
        
        # Compute conformity score on calibration set and on new data example
        pred_low_for_cal = res_lower.predict(X_cal)
        pred_up_for_cal = res_upper.predict(X_cal)
        if len(X_train.shape) == 1:
            scores = np.maximum(Y_cal - pred_up_for_cal, pred_low_for_cal - Y_cal)
        else:
            scores = np.maximum( np.linalg.norm(Y_cal - pred_up_for_cal, axis=1), np.linalg.norm(pred_low_for_cal - Y_cal, axis=1))
        q_up = res_upper.predict(X[t].reshape(1, -1))[0]
        q_low = res_lower.predict(X[t].reshape(1, -1))[0]
        if len(X_train.shape) == 1:
            new_score = max(Y[t] - q_up, q_low - Y[t])
        else:
            new_score = np.maximum( np.linalg.norm(Y[t] - q_up), np.linalg.norm(q_low - Y[t]))
        
        # Compute errt for both methods
        conf_quant_naive = np.quantile(scores, 1-alpha)
        no_adapt_error_seq[t-t_init] = float(conf_quant_naive < new_score)
        band_native.append(conf_quant_naive)
        
        if alpha_t >= 1:
            adapt_err_seq[t-t_init] = 1
            band_adapt.append(0)
        elif alpha_t <= 0:
            adapt_err_seq[t-t_init] = 0
            band_adapt.append(max_band)
        else:
            conf_quant_adapt = np.quantile(scores, 1-alpha_t)
            adapt_err_seq[t-t_init] = float(conf_quant_adapt < new_score)
            band_adapt.append(conf_quant_adapt)
        # update alpha_t
        alpha_trajectory[t-t_init] = alpha_t
        if update_method == "Simple":
            alpha_t += gamma * (alpha - adapt_err_seq[t-t_init])
        elif update_method == "Momentum":
            w = momentum_bw ** np.arange(t-t_init+1)[::-1]
            w /= w.sum()
            alpha_t += gamma * (alpha - np.sum(adapt_err_seq[:t-t_init+1] * w))
        
        # if t % 100 == 0:
        #     print(f"Done {t} time steps")
    
    return alpha_trajectory, adapt_err_seq, no_adapt_error_seq, (band_native, band_adapt)

