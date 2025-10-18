#!/usr/bin/env python3
"""
Comprehensive baseline evaluation script for conformal prediction methods.
Runs all baselines on datasets in data/inference_per_z/ and generates comparison table.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import baseline methods
from algos.aci import ACI
from algos.cptc import conformal_prediction_cptc
from algos.agaci import agaci
from algos.dtaci import dtaci
from algos.mvp import mvp
from algos.pid_eci import ECI, OGD, decay_OGD
#import algos.SPCI_class as SPCI
from sklearn.ensemble import RandomForestRegressor
import torch


class DataLoader:
    """Handles loading and preprocessing of datasets"""

    def __init__(self, data_dir="data/inference_per_z"):
        self.data_dir = data_dir

    def load_dataset(self, filename):
        """
        Load dataset and return standardized format

        Returns:
            dict with keys: ground_truth, predictions, z_probs, dataset_name, format_type
        """
        filepath = os.path.join(self.data_dir, filename)
        dataset_name = filename.replace('.npz', '')

        data = np.load(filepath)

        return self._load_data_standard_format(data, dataset_name)


    def _load_data_standard_format(self, data, dataset_name):
        """Load GluonTS dataset format (electricity, traffic)"""
        ground_truth = data['ground_truth']  # (n_series, pred_length) or (n_series, pred_length, n_dims)
        predictions = data['all_mean']       # (n_series, pred_length) or (n_series, pred_length, n_dims)
        z_probs = data['all_z_probs']       # (n_series, pred_length, n_states)
        lower_bound = data['all_lb']        # (n_series, pred_length) or (n_series, pred_length, n_dims)
        upper_bound = data['all_ub']        # (n_series, pred_length) or (n_series, pred_length, n_dims)
        z_mean = data['z_mean']            # (n_series, pred_length, n_states, n_dims)

        # Only squeeze if last dimension is 1 (univariate stored as 3D)
        if len(ground_truth.shape) == 3 and ground_truth.shape[-1] == 1:
            ground_truth = ground_truth.squeeze(-1)
        if len(predictions.shape) == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        if len(lower_bound.shape) == 3 and lower_bound.shape[-1] == 1:
            lower_bound = lower_bound.squeeze(-1)
        if len(upper_bound.shape) == 3 and upper_bound.shape[-1] == 1:
            upper_bound = upper_bound.squeeze(-1)
        if len(z_mean.shape) == 3 and z_mean.shape[-1] == 1:
            z_mean = z_mean.squeeze(-1)

        return {
            'ground_truth': ground_truth,
            'predictions': predictions,
            'z_probs': z_probs,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'dataset_name': dataset_name,
            'z_mean': z_mean,
            'format_type': 'gluonts'
        }

    def get_all_datasets(self, exclude=[]):
        """Get list of all dataset files, excluding specified ones"""
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        return [f for f in all_files if not any(ex in f for ex in exclude)]


class BaselineRunner:
    """Runs baseline conformal prediction methods"""

    def __init__(self, alpha=0.1, gamma=0.01, results_dir="results/new"):
        self.alpha = alpha
        self.gamma = gamma
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)

        self.max_widths = {"bee": 0.5,
        "bouncing_ball_obs": 5,
        "bouncing_ball_dyn": 3,
        "3_mode_system": 3,
        "traffic": 0.2,
        "electricity": 60
        }

        self.cptc_gammas = {"bee": 3,
        "bouncing_ball_obs": 1,
        "bouncing_ball_dyn": 0.2,
        "3_mode_system": 0.2,
        "traffic": 1,
        "electricity": 1
        }

    def compute_warm_start(self, pred_len):
        """Calculate warm start period matching existing scripts"""
        return min(int(pred_len / 2), 100)

    def compute_conformity_scores(self, ground_truth, predictions, dataset_name):
        """
        Compute conformity scores handling both univariate and multivariate data

        For multivariate data (e.g., Bee dataset), computes L2 distance using first 2 dimensions
        For univariate data, computes absolute error

        Args:
            ground_truth: Shape (n_series, time_steps) or (n_series, time_steps, n_dims)
            predictions: Shape (n_series, time_steps) or (n_series, time_steps, n_dims)
            dataset_name: Name of dataset

        Returns:
            scores: Shape (n_series, time_steps) - conformity scores
        """
        # Check if data is multivariate
        if len(ground_truth.shape) == 3 and ground_truth.shape[-1] > 1:
            # Multivariate: use L2 distance on first 2 dimensions
            gt_subset = ground_truth[:, :, :2]  # (n_series, time_steps, 2)
            pred_subset = predictions[:, :, :2]  # (n_series, time_steps, 2)
            scores = np.linalg.norm((gt_subset - pred_subset), axis=-1)  # (n_series, time_steps)
        else:
            # Univariate: use absolute error
            scores = np.abs(ground_truth - predictions)

        return scores

    def run_cp(self, dataset_dict):
        """Run standard conformal prediction (CP) baseline"""
        print(f"\nRunning CP on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]
        warm_start = self.compute_warm_start(pred_len)

        # Compute conformity scores for all series
        all_scores = self.compute_conformity_scores(
            ground_truth[:, -pred_len:],
            predictions,
            dataset_dict['dataset_name']
        )

        all_coverages = []
        all_widths = []

        for ts in tqdm(range(ground_truth.shape[0]), desc="CP"):
            scores_ts = all_scores[ts]

            # Warm start calibration
            scores = scores_ts[:warm_start]
            width = []
            coverage = []

            # Online conformal prediction
            for t in range(warm_start, len(scores_ts)):
                score_t = scores_ts[t]
                thresh = np.quantile(scores, 1 - self.alpha)
                width.append(thresh)
                coverage.append(score_t <= thresh)
                scores = np.append(scores, score_t)

            all_coverages.append(coverage)
            all_widths.append(width)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_cp.npz")
        np.savez(save_path, all_coverages=all_coverages, all_widths=all_widths)

        return self._compute_metrics(all_coverages, all_widths, dataset_dict['dataset_name'])

    def run_REDSDS(self, dataset_dict):
        """Run REDSDS baseline using model's prediction intervals (90% density)"""
        print(f"\nRunning REDSDS on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        lower_bound = dataset_dict['lower_bound']
        upper_bound = dataset_dict['upper_bound']
        pred_len = lower_bound.shape[1]

        all_coverages = []
        all_widths = []

        # Check if multivariate
        is_multivariate = len(ground_truth.shape) == 3 and ground_truth.shape[-1] > 1

        for ts in tqdm(range(ground_truth.shape[0]), desc="REDSDS"):
            gt = ground_truth[ts, -pred_len:]
            lb = lower_bound[ts]
            ub = upper_bound[ts]
            pred = predictions[ts]

            if is_multivariate:
                # For multivariate: check if each dimension (first 2) is within bounds
                # and compute average width across dimensions
                gt_subset = gt[:, :2]  # (time_steps, 2)
                lb_subset = lb[:, :2]  # (time_steps, 2)
                ub_subset = ub[:, :2]  # (time_steps, 2)

                # Coverage: both dimensions must be within bounds
                coverage_per_dim = ((gt_subset >= lb_subset) & (gt_subset <= ub_subset))  # (time_steps, 2)
                coverage = np.all(coverage_per_dim, axis=-1).astype(int)  # (time_steps,)

                # Width: average radius across first 2 dimensions
                width_per_dim = (ub_subset - lb_subset) / 2  # (time_steps, 2)
                width = np.mean(width_per_dim, axis=-1)  # (time_steps,)
            else:
                # Univariate: original logic
                coverage = ((gt >= lb) & (gt <= ub)).astype(int)
                width = (ub - lb) / 2

            all_coverages.append(coverage)
            all_widths.append(width)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_REDSDS.npz")
        np.savez(save_path, all_coverages=all_coverages, all_widths=all_widths)

        return self._compute_metrics(all_coverages, all_widths, dataset_dict['dataset_name'])

    def run_aci(self, dataset_dict):
        """Run Adaptive Conformal Inference (ACI) baseline"""
        print(f"\nRunning ACI on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]
        warm_start = self.compute_warm_start(pred_len)

        # Check if multivariate
        is_multivariate = len(ground_truth.shape) == 3 and ground_truth.shape[-1] > 1

        if is_multivariate:
            # For multivariate: compute conformity scores and use X=0 (dummy predictor)
            all_scores = self.compute_conformity_scores(
                ground_truth[:, -pred_len:],
                predictions,
                dataset_dict['dataset_name']
            )
            # Use zeros as X since we're directly using scores as Y
            Xs = np.zeros_like(all_scores)
            Ys = all_scores
        else:
            # Univariate: use original X, Y format
            Xs, Ys = self._create_x_y_from_pred(predictions, ground_truth[:, -pred_len:])

        alpha_trajectories = []
        adapt_err_seqs = []
        no_adapt_error_seqs = []
        band_natives = []
        band_adapts = []

        for i in tqdm(range(Xs.shape[0]), desc="ACI"):
            X, Y = Xs[i], Ys[i]
            result = ACI(Y, X, alpha=self.alpha, gamma=self.gamma, t_init=warm_start)
            alpha_trajectory, adapt_err_seq, no_adapt_error_seq, (band_native, band_adapt) = result

            alpha_trajectories.append(alpha_trajectory)
            adapt_err_seqs.append(adapt_err_seq)
            no_adapt_error_seqs.append(no_adapt_error_seq)
            band_natives.append(band_native)
            band_adapts.append(band_adapt)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_aci.npz")
        np.savez(save_path,
                 alpha_trajectories=alpha_trajectories,
                 adapt_err_seqs=adapt_err_seqs,
                 no_adapt_error_seqs=no_adapt_error_seqs,
                 band_natives=band_natives,
                 band_adapts=band_adapts)

        # Compute metrics from ACI results
        coverage = (1 - np.mean(adapt_err_seqs)) * 100
        cov_std = np.std(np.mean(adapt_err_seqs, axis=1)) * 100
        width = np.mean(band_adapts) * 2
        width_std = np.std(np.mean(band_adapts, axis=1)) * 2

        return coverage, cov_std, width, width_std

    def run_cptc(self, dataset_dict):
        """Run Conformal Prediction for Time series with Change Points (CPTC)"""
        print(f"\nRunning CPTC on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        z_probs = dataset_dict['z_probs']
        z_means = dataset_dict['z_mean']
        max_width = self.max_widths[dataset_dict['dataset_name']]
        gamma = self.cptc_gammas[dataset_dict['dataset_name']]
        
        pred_len = predictions.shape[1]
        warm_start = self.compute_warm_start(pred_len)

        # Check if multivariate
        is_multivariate = len(ground_truth.shape) == 3 and ground_truth.shape[-1] > 1
        all_coverages = []
        all_widths = []

        for i in tqdm(range(ground_truth.shape[0]), desc="CPTC"):
            gt = ground_truth[i, -pred_len:]
            pred = predictions[i]
            z_prob = z_probs[i]
            z_mean = z_means[i]

            if is_multivariate:
                # For multivariate: compute conformity scores first
                # Create temporary arrays for this series
                gt_expanded = gt[np.newaxis, :]  # (1, pred_len, n_dims)
                pred_expanded = pred[np.newaxis, :]  # (1, pred_len, n_dims)
                scores = self.compute_conformity_scores(gt_expanded, pred_expanded, dataset_dict['dataset_name'])[0]

                # Handle case where z_probs length < pred_len
                if z_prob.shape[0] < pred_len:
                    effective_warm_start = min(warm_start, z_prob.shape[0] // 2)
                    scores_subset = scores[:z_prob.shape[0]]
                    # Pass scores as gt, zeros as pred (CPTC will compute residuals)
                    coverages, widths, _ = conformal_prediction_cptc(
                          gt[:z_prob.shape[0]], pred[:z_prob.shape[0]], z_prob, z_mean,
                        T=effective_warm_start, alpha=self.alpha, gamma=gamma, max_width=max_width
                    )
                else:
                    coverages, widths, _ = conformal_prediction_cptc(
                        gt[:z_prob.shape[0]], pred[:z_prob.shape[0]], z_prob, z_mean,
                        T=warm_start, alpha=self.alpha, gamma=gamma, max_width=max_width
                    )
            else:
                # Univariate: original logic
                if z_prob.shape[0] < pred_len:
                    effective_warm_start = min(warm_start, z_prob.shape[0] // 2)
                    coverages, widths, _ = conformal_prediction_cptc(
                        gt[:z_prob.shape[0]], pred[:z_prob.shape[0]], z_prob, z_mean,
                        T=effective_warm_start, alpha=self.alpha, gamma=gamma, max_width=max_width
                    )
                else:
                    coverages, widths, _ = conformal_prediction_cptc(
                        gt, pred, z_prob, z_mean,
                        T=warm_start, alpha=self.alpha, gamma=gamma, max_width=max_width
                    )

            all_coverages.append(coverages)
            all_widths.append(widths)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_cptc.npz")
        np.savez(save_path, all_coverages=all_coverages, all_widths=all_widths)

        return self._compute_metrics(all_coverages, all_widths, dataset_dict['dataset_name'])

    def run_agaci(self, dataset_dict):
        """Run Aggregated Adaptive Conformal Inference (AgACI)"""
        print(f"\nRunning AgACI on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]

        # Compute conformity scores
        scores = self.compute_conformity_scores(
            ground_truth[:, -pred_len:],
            predictions,
            dataset_dict['dataset_name']
        )

        all_alpha_seqs = []
        all_err_seqs = []
        all_gamma_seqs = []

        gammas = [0.001, 0.01, 0.1]

        for i in tqdm(range(scores.shape[0]), desc="AgACI"):
            alpha_seq, err_seq, gamma_seq = agaci(scores[i], alpha=self.alpha, gammas=gammas)
            all_alpha_seqs.append(alpha_seq)
            all_err_seqs.append(err_seq)
            all_gamma_seqs.append(gamma_seq)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_agaci.npz")
        np.savez(save_path, alpha_seqs=all_alpha_seqs, err_seqs=all_err_seqs, gamma_seqs=all_gamma_seqs)

        # Compute metrics: use alpha_seq as width, err_seq for coverage
        coverage = (1 - np.mean(all_err_seqs)) * 100
        cov_std = np.std(np.mean(all_err_seqs, axis=1)) * 100
        width = np.mean(all_alpha_seqs) * 2
        width_std = np.std(np.mean(all_alpha_seqs, axis=1)) * 2

        if 'traffic' in dataset_dict['dataset_name']:
            width = width * 100
            width_std = width_std * 100

        return coverage, cov_std, width, width_std

    def run_dtaci(self, dataset_dict):
        """Run Distribution-free Time-series Adaptive Conformal Inference (DtACI)"""
        print(f"\nRunning DtACI on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]

        # Compute conformity scores
        scores = self.compute_conformity_scores(
            ground_truth[:, -pred_len:],
            predictions,
            dataset_dict['dataset_name']
        )

        all_alpha_seqs = []
        all_err_seqs = []
        all_mean_alpha_seqs = []

        gammas = [0.001, 0.01, 0.1]

        for i in tqdm(range(scores.shape[0]), desc="DtACI"):
            results = dtaci(scores[i], alpha=self.alpha, gammas=gammas)
            alpha_seq, err_seq_adapt, err_seq_fixed, gamma_seq, mean_alpha_seq, mean_err_seq, mean_gammas = results
            all_alpha_seqs.append(alpha_seq)
            all_err_seqs.append(err_seq_adapt)
            all_mean_alpha_seqs.append(mean_alpha_seq)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_dtaci.npz")
        np.savez(save_path, alpha_seqs=all_alpha_seqs, err_seqs=all_err_seqs, mean_alpha_seqs=all_mean_alpha_seqs)

        # Compute metrics
        coverage = (1 - np.mean(all_err_seqs)) * 100
        cov_std = np.std(np.mean(all_err_seqs, axis=1)) * 100
        width = np.mean(all_alpha_seqs) * 2
        width_std = np.std(np.mean(all_alpha_seqs, axis=1)) * 2

        if 'traffic' in dataset_dict['dataset_name']:
            width = width * 100
            width_std = width_std * 100

        return coverage, cov_std, width, width_std

    def run_mvp(self, dataset_dict):
        """Run Multi-Valid Prediction (MVP)"""
        print(f"\nRunning MVP on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]

        # Compute conformity scores
        scores = self.compute_conformity_scores(
            ground_truth[:, -pred_len:],
            predictions,
            dataset_dict['dataset_name']
        )
        # Normalize scores to [0, 1] range
        scores_min = scores.min(axis=1, keepdims=True)
        scores_max = scores.max(axis=1, keepdims=True)
        scores_normalized = (scores - scores_min) / (scores_max - scores_min + 1e-10)

        all_alpha_seqs = []
        all_err_seqs = []

        for i in tqdm(range(scores_normalized.shape[0]), desc="MVP"):
            alpha_seq, err_seq = mvp(scores_normalized[i], alpha=self.alpha, m=40)
            all_alpha_seqs.append(alpha_seq)
            all_err_seqs.append(err_seq)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_mvp.npz")
        np.savez(save_path, alpha_seqs=all_alpha_seqs, err_seqs=all_err_seqs)

        # Compute metrics
        coverage = (1 - np.mean(all_err_seqs)) * 100
        cov_std = np.std(np.mean(all_err_seqs, axis=1)) * 100
        # For MVP, alpha_seq is the threshold, need to convert back to original scale
        width = np.mean([np.mean(alpha_seq) * (scores_max[i] - scores_min[i])
                        for i, alpha_seq in enumerate(all_alpha_seqs)]) * 2
        width_std = np.std([np.mean(alpha_seq) * (scores_max[i] - scores_min[i])
                           for i, alpha_seq in enumerate(all_alpha_seqs)]) * 2

        if 'traffic' in dataset_dict['dataset_name']:
            width = width * 100
            width_std = width_std * 100

        return coverage, cov_std, width, width_std

    def run_eci(self, dataset_dict):
        """Run Error Correction with Integrator (ECI)"""
        print(f"\nRunning ECI on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]
        warm_start = self.compute_warm_start(pred_len)

        # Compute conformity scores
        scores = self.compute_conformity_scores(
            ground_truth[:, -pred_len:],
            predictions,
            dataset_dict['dataset_name']
        )

        all_qs = []
        all_coverages = []

        for i in tqdm(range(scores.shape[0]), desc="ECI"):
            result = ECI(scores[i], alpha=self.alpha, lr=0.01, T_burnin=warm_start, ahead=1)
            qs = result['q']
            coverage = (scores[i] <= qs).astype(int)
            all_qs.append(qs)
            all_coverages.append(coverage)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_eci.npz")
        np.savez(save_path, qs=all_qs, coverages=all_coverages)

        # Compute metrics
        coverage_pct = np.mean(all_coverages) * 100
        cov_std = np.std(np.mean(all_coverages, axis=1)) * 100
        width = np.mean(all_qs) * 2
        width_std = np.std(np.mean(all_qs, axis=1)) * 2

        if 'traffic' in dataset_dict['dataset_name']:
            width = width * 100
            width_std = width_std * 100

        return coverage_pct, cov_std, width, width_std

    def run_ogd(self, dataset_dict):
        """Run Online Gradient Descent (OGD)"""
        print(f"\nRunning OGD on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]

        # Compute conformity scores
        scores = self.compute_conformity_scores(
            ground_truth[:, -pred_len:],
            predictions,
            dataset_dict['dataset_name']
        )

        all_qs = []
        all_coverages = []

        for i in tqdm(range(scores.shape[0]), desc="OGD"):
            result = OGD(scores[i], alpha=self.alpha, lr=0.01, ahead=1)
            qs = result['q']
            coverage = (scores[i] <= qs).astype(int)
            all_qs.append(qs)
            all_coverages.append(coverage)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_ogd.npz")
        np.savez(save_path, qs=all_qs, coverages=all_coverages)

        # Compute metrics
        coverage_pct = np.mean(all_coverages) * 100
        cov_std = np.std(np.mean(all_coverages, axis=1)) * 100
        width = np.mean(all_qs) * 2
        width_std = np.std(np.mean(all_qs, axis=1)) * 2

        if 'traffic' in dataset_dict['dataset_name']:
            width = width * 100
            width_std = width_std * 100

        return coverage_pct, cov_std, width, width_std

    def run_decay_ogd(self, dataset_dict):
        """Run Decaying Online Gradient Descent (decay_OGD)"""
        print(f"\nRunning decay_OGD on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]

        # Compute conformity scores
        scores = self.compute_conformity_scores(
            ground_truth[:, -pred_len:],
            predictions,
            dataset_dict['dataset_name']
        )

        all_qs = []
        all_coverages = []

        for i in tqdm(range(scores.shape[0]), desc="decay_OGD"):
            result = decay_OGD(scores[i], alpha=self.alpha, lr=0.01, ahead=1)
            qs = result['q']
            coverage = (scores[i] <= qs).astype(int)
            all_qs.append(qs)
            all_coverages.append(coverage)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_decay_ogd.npz")
        np.savez(save_path, qs=all_qs, coverages=all_coverages)

        # Compute metrics
        coverage_pct = np.mean(all_coverages) * 100
        cov_std = np.std(np.mean(all_coverages, axis=1)) * 100
        width = np.mean(all_qs) * 2
        width_std = np.std(np.mean(all_qs, axis=1)) * 2

        if 'traffic' in dataset_dict['dataset_name']:
            width = width * 100
            width_std = width_std * 100

        return coverage_pct, cov_std, width, width_std

    # def run_enbpi(self, dataset_dict, past_window=30):
    #     """Run Ensemble Bootstrap Prediction Intervals (EnbPI)"""
    #     print(f"\nRunning EnbPI on {dataset_dict['dataset_name']}...")

    #     ground_truth = dataset_dict['ground_truth']
    #     predictions = dataset_dict['predictions']
    #     pred_len = predictions.shape[1]
    #     warm_start = self.compute_warm_start(pred_len)

    #     # Check if multivariate - EnbPI works best with univariate
    #     is_multivariate = len(ground_truth.shape) == 3 and ground_truth.shape[-1] > 1
    #     if is_multivariate:
    #         print("  Warning: EnbPI designed for univariate data, using first dimension")
    #         ground_truth = ground_truth[:, :, 0]
    #         predictions = predictions[:, :, 0]

    #     all_coverages = []
    #     all_widths = []

    #     for sample_i in tqdm(range(ground_truth.shape[0]), desc="EnbPI"):
    #         # Create sliding window features (window size = 2)
    #         window = 2
    #         gt_full = ground_truth[sample_i]
    #         pred_full = predictions[sample_i]

    #         # Create features: X[t] = [pred[t], pred[t+1]], Y[t] = gt[t+2]
    #         X_full = torch.tensor([pred_full[a:a+window] for a in range(pred_len-window)])
    #         Y_full = torch.tensor(gt_full[-pred_len+window:], dtype=torch.float32)

    #         # Split into train and predict
    #         N = warm_start
    #         X_train = X_full[:N]
    #         X_predict = X_full[N:]
    #         Y_train = Y_full[:N]
    #         Y_predict = Y_full[N:]

    #         # Build EnbPI model
    #         fit_func = RandomForestRegressor(
    #             n_estimators=10, max_depth=1, criterion='squared_error',
    #             bootstrap=False, n_jobs=-1, random_state=1103
    #         )

    #         enbpi_model = SPCI.SPCI_and_EnbPI(X_train, X_predict, Y_train, Y_predict, fit_func=fit_func)

    #         # Fit bootstrap models
    #         enbpi_model.fit_bootstrap_models_online_multistep(B=25, fit_sigmaX=False, stride=1)

    #         # Compute prediction intervals (use_SPCI=False for EnbPI)
    #         enbpi_model.compute_PIs_Ensemble_online(
    #             self.alpha, smallT=True, past_window=past_window,
    #             use_SPCI=False, quantile_regr=False, stride=1
    #         )

    #         # Get results
    #         results = enbpi_model.get_results(self.alpha, 'data', 1)
    #         coverage = results['coverage'].item()
    #         width = results['width'].item()

    #         all_coverages.append(coverage)
    #         all_widths.append(width)

    #     # Save results
    #     save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_enbpi.npz")
    #     np.savez(save_path, all_coverages=all_coverages, all_widths=all_widths)

    #     # Compute metrics
    #     coverage_pct = np.mean(all_coverages) * 100
    #     cov_std = np.std(all_coverages) * 100
    #     width_mean = np.mean(all_widths)
    #     width_std = np.std(all_widths)

    #     if 'traffic' in dataset_dict['dataset_name']:
    #         width_mean = width_mean * 100
    #         width_std = width_std * 100

    #     return coverage_pct, cov_std, width_mean, width_std

    def run_spci(self, dataset_dict, past_window=1):
        """Run Split Conformal Prediction Intervals (SPCI) with quantile regression"""
        print(f"\nRunning SPCI on {dataset_dict['dataset_name']}...")

        ground_truth = dataset_dict['ground_truth']
        predictions = dataset_dict['predictions']
        pred_len = predictions.shape[1]
        warm_start = self.compute_warm_start(pred_len)

        # Check if multivariate - SPCI works best with univariate
        is_multivariate = len(ground_truth.shape) == 3 and ground_truth.shape[-1] > 1
        if is_multivariate:
            print("  Warning: SPCI designed for univariate data, using first dimension")
            ground_truth = ground_truth[:, :, 0]
            predictions = predictions[:, :, 0]

        all_coverages = []
        all_widths = []

        for sample_i in tqdm(range(ground_truth.shape[0]), desc="SPCI"):
            # Create sliding window features (window size = 2)
            window = 2
            gt_full = ground_truth[sample_i]
            pred_full = predictions[sample_i]

            # Create features: X[t] = [pred[t], pred[t+1]], Y[t] = gt[t+2]
            X_full = torch.tensor([pred_full[a:a+window] for a in range(pred_len-window)])
            Y_full = torch.tensor(gt_full[-pred_len+window:], dtype=torch.float32)

            # Split into train and predict
            N = warm_start
            X_train = X_full[:N]
            X_predict = X_full[N:]
            Y_train = Y_full[:N]
            Y_predict = Y_full[N:]

            # Build SPCI model
            fit_func = RandomForestRegressor(
                n_estimators=10, max_depth=1, criterion='squared_error',
                bootstrap=False, n_jobs=-1, random_state=1103
            )

            spci_model = SPCI.SPCI_and_EnbPI(X_train, X_predict, Y_train, Y_predict, fit_func=fit_func)

            # Fit bootstrap models
            spci_model.fit_bootstrap_models_online_multistep(B=25, fit_sigmaX=False, stride=1)

            # Compute prediction intervals (use_SPCI=True for SPCI with quantile regression)
            spci_model.compute_PIs_Ensemble_online(
                self.alpha, smallT=False, past_window=past_window,
                use_SPCI=True, quantile_regr=True, stride=1
            )

            # Get results
            results = spci_model.get_results(self.alpha, 'data', 1)
            coverage = results['coverage'].item()
            width = results['width'].item()

            all_coverages.append(coverage)
            all_widths.append(width)

        # Save results
        save_path = os.path.join(self.results_dir, f"{dataset_dict['dataset_name']}_spci.npz")
        np.savez(save_path, all_coverages=all_coverages, all_widths=all_widths)

        # Compute metrics
        coverage_pct = np.mean(all_coverages) * 100
        cov_std = np.std(all_coverages) * 100
        width_mean = np.mean(all_widths)
        width_std = np.std(all_widths)

        if 'traffic' in dataset_dict['dataset_name']:
            width_mean = width_mean * 100
            width_std = width_std * 100

        return coverage_pct, cov_std, width_mean, width_std

    def _create_x_y_from_pred(self, preds, gts):
        """Create X, Y arrays from predictions and ground truth for ACI"""
        Xs = []
        Ys = []
        for i in range(len(preds)):
            X, Y = [], []
            for t in range(len(preds[i])):
                X.append(preds[i, t])
                Y.append(gts[i, t])
            Xs.append(X)
            Ys.append(Y)
        return np.array(Xs), np.array(Ys)

    def _compute_metrics(self, all_coverages, all_widths, dataset_name):
        """Compute coverage and width statistics"""
        coverage = np.mean(all_coverages) * 100
        cov_std = np.std(np.mean(all_coverages, axis=1)) * 100
        width = np.mean(all_widths) * 2  # Diameter
        width_std = np.std(np.mean(all_widths, axis=1)) * 2

        # Apply traffic scaling
        if 'traffic' in dataset_name:
            width = width * 100
            width_std = width_std * 100

        return coverage, cov_std, width, width_std


class ResultsTable:
    """Generates formatted results table"""

    def __init__(self):
        self.results = {}

    def add_result(self, dataset, method, coverage, cov_std, width, width_std):
        """Add a result to the table"""
        if dataset not in self.results:
            self.results[dataset] = {}
        self.results[dataset][method] = {
            'coverage': coverage,
            'cov_std': cov_std,
            'width': width,
            'width_std': width_std
        }

    def print_table(self):
        """Print results in LaTeX table format"""
        print("\n" + "="*80)
        print("RESULTS TABLE")
        print("="*80)

        # Get all methods and datasets
        datasets = sorted(self.results.keys())
        methods = set()
        for dataset in datasets:
            methods.update(self.results[dataset].keys())
        methods = sorted(methods)

        # Print header
        print(f"\n{'Dataset':<25} {'Method':<10} {'Coverage (%)':<20} {'Width':<20}")
        print("-" * 80)

        # Print results
        for dataset in datasets:
            for method in methods:
                if method in self.results[dataset]:
                    r = self.results[dataset][method]
                    cov_str = f"{r['coverage']:.2f} ± {r['cov_std']:.2f}"
                    width_str = f"{r['width']:.2f} ± {r['width_std']:.2f}"
                    print(f"{dataset:<25} {method:<10} {cov_str:<20} {width_str:<20}")
            print()

    def save_csv(self, filepath):
        """Save results table to CSV file"""
        import pandas as pd

        # Prepare data for CSV
        rows = []
        for dataset in sorted(self.results.keys()):
            for method in sorted(self.results[dataset].keys()):
                r = self.results[dataset][method]
                rows.append({
                    'Dataset': dataset,
                    'Method': method,
                    'Coverage': r['coverage'],
                    'Coverage_Std': r['cov_std'],
                    'Width': r['width'],
                    'Width_Std': r['width_std']
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")


def load_and_generate_table(results_dir="results/new", output_csv="results_table.csv"):
    """
    Load saved results from results_dir and generate a comparison table

    Args:
        results_dir: Directory containing saved .npz result files
        output_csv: Path to save CSV output
    """
    import glob

    table = ResultsTable()
    result_files = glob.glob(os.path.join(results_dir, "*.npz"))

    print(f"Found {len(result_files)} result files in {results_dir}")

    for filepath in result_files:
        filename = os.path.basename(filepath)
        # Parse filename: {dataset}_{method}.npz
        parts = filename.replace('.npz', '').rsplit('_', 1)
        if len(parts) != 2:
            print(f"Skipping {filename}: unexpected format")
            continue

        dataset_name, method = parts[0], parts[1].upper()

        # Load data
        data = np.load(filepath)

        # Compute metrics based on method type
        try:
            if method in ['CP']:
                # These have all_coverages and all_widths
                all_coverages = data['all_coverages']
                all_widths = data['all_widths']

                coverage = np.mean(all_coverages) * 100
                cov_std = np.std(np.mean(all_coverages, axis=1)) * 100
                width = np.mean(all_widths) * 2  # Diameter
                width_std = np.std(np.mean(all_widths, axis=1)) * 2

            elif method == 'ACI':
                # Has band_adapts and adapt_err_seqs
                adapt_err_seqs = data['adapt_err_seqs']
                band_adapts = data['band_adapts']

                coverage = (1 - np.mean(adapt_err_seqs)) * 100
                cov_std = np.std(np.mean(adapt_err_seqs, axis=1)) * 100
                width = np.mean(band_adapts) * 2
                width_std = np.std(np.mean(band_adapts, axis=1)) * 2

            elif method in ['AGACI', 'DTACI', 'MVP']:
                # These have alpha_seqs and err_seqs
                alpha_seqs = data['alpha_seqs']
                err_seqs = data['err_seqs']

                coverage = (1 - np.mean(err_seqs)) * 100
                cov_std = np.std(np.mean(err_seqs, axis=1)) * 100
                width = np.mean(alpha_seqs) * 2
                width_std = np.std(np.mean(alpha_seqs, axis=1)) * 2

            elif method in ['ECI', 'OGD', 'DECAY_OGD']:
                # These have qs and coverages
                all_qs = data['qs']
                all_coverages = data['coverages']

                coverage = np.mean(all_coverages) * 100
                cov_std = np.std(np.mean(all_coverages, axis=1)) * 100
                width = np.mean(all_qs) * 2
                width_std = np.std(np.mean(all_qs, axis=1)) * 2

            elif method in ['ENBPI', 'SPCI', 'CPTC', 'REDSDS']:
                # These have all_coverages and all_widths (already computed as widths, not radius)
                all_coverages = data['all_coverages']
                all_widths = data['all_widths']

                coverage = np.mean(all_coverages) * 100
                cov_std = np.std(all_coverages) * 100
                width = np.mean(all_widths)  # Already width, not radius
                width_std = np.std(all_widths)

            else:
                print(f"Unknown method {method} in {filename}")
                continue

            # # Apply traffic scaling
            if 'traffic' in dataset_name:
                width = width * 100
                width_std = width_std * 100

            table.add_result(dataset_name, method, coverage, cov_std, width, width_std)
            print(f"Loaded {dataset_name} - {method}: Coverage={coverage:.2f}%, Width={width:.2f}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Print and save table
    table.print_table()
    table.save_csv(output_csv)

    return table


def main():
    parser = argparse.ArgumentParser(description='Run conformal prediction baselines')
    parser.add_argument('--methods', nargs='+', default=['REDSDS', 'CP', 'ACI', 'CPTC'],
                       help='Methods to run: REDSDS, CP, ACI, CPTC, AgACI, MVP, ECI, OGD, decay_OGD, EnbPI, SPCI')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to run (default: all)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Miscoverage rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.03,
                       help='Learning rate for adaptive methods (default: 0.01)')
    parser.add_argument('--load-results', action='store_true',
                       help='Load saved results and generate table instead of running experiments')
    parser.add_argument('--results-dir', type=str, default='results/new',
                       help='Directory containing saved results (default: results/new)')
    parser.add_argument('--output-csv', type=str, default='results_table.csv',
                       help='Output CSV file path (default: results_table.csv)')
    args = parser.parse_args()

    # If load-results flag is set, just load and generate table
    if args.load_results:
        load_and_generate_table(args.results_dir, args.output_csv)
        return

    # Initialize components
    loader = DataLoader()
    runner = BaselineRunner(alpha=args.alpha, gamma=args.gamma)
    table = ResultsTable()

    # Get datasets
    if args.datasets:
        dataset_files = [f"{d}.npz" if not d.endswith('.npz') else d for d in args.datasets]
    else:
        dataset_files = loader.get_all_datasets(exclude=[])

    print(f"Running methods {args.methods} on datasets: {dataset_files}")

    # Run baselines on each dataset
    for dataset_file in dataset_files:
        dataset_dict = loader.load_dataset(dataset_file)
        dataset_name = dataset_dict['dataset_name']

        # Run selected methods
        if 'REDSDS' in args.methods:
            cov, cov_std, width, width_std = runner.run_REDSDS(dataset_dict)
            table.add_result(dataset_name, 'REDSDS', cov, cov_std, width, width_std)

        if 'CP' in args.methods:
            cov, cov_std, width, width_std = runner.run_cp(dataset_dict)
            table.add_result(dataset_name, 'CP', cov, cov_std, width, width_std)

        if 'ACI' in args.methods:
            cov, cov_std, width, width_std = runner.run_aci(dataset_dict)
            table.add_result(dataset_name, 'ACI', cov, cov_std, width, width_std)

        if 'CPTC' in args.methods:
            cov, cov_std, width, width_std = runner.run_cptc(dataset_dict)
            table.add_result(dataset_name, 'CPTC', cov, cov_std, width, width_std)

        if 'AgACI' in args.methods:
            cov, cov_std, width, width_std = runner.run_agaci(dataset_dict)
            table.add_result(dataset_name, 'AgACI', cov, cov_std, width, width_std)

        if 'DtACI' in args.methods:
            cov, cov_std, width, width_std = runner.run_dtaci(dataset_dict)
            table.add_result(dataset_name, 'DtACI', cov, cov_std, width, width_std)

        if 'MVP' in args.methods:
            cov, cov_std, width, width_std = runner.run_mvp(dataset_dict)
            table.add_result(dataset_name, 'MVP', cov, cov_std, width, width_std)

        if 'ECI' in args.methods:
            cov, cov_std, width, width_std = runner.run_eci(dataset_dict)
            table.add_result(dataset_name, 'ECI', cov, cov_std, width, width_std)

        if 'OGD' in args.methods:
            cov, cov_std, width, width_std = runner.run_ogd(dataset_dict)
            table.add_result(dataset_name, 'OGD', cov, cov_std, width, width_std)

        if 'decay_OGD' in args.methods:
            cov, cov_std, width, width_std = runner.run_decay_ogd(dataset_dict)
            table.add_result(dataset_name, 'decay_OGD', cov, cov_std, width, width_std)

        if 'EnbPI' in args.methods:
            cov, cov_std, width, width_std = runner.run_enbpi(dataset_dict)
            table.add_result(dataset_name, 'EnbPI', cov, cov_std, width, width_std)

        if 'SPCI' in args.methods:
            cov, cov_std, width, width_std = runner.run_spci(dataset_dict)
            table.add_result(dataset_name, 'SPCI', cov, cov_std, width, width_std)

    # Print results table
    table.print_table()


if __name__ == "__main__":
    main()
