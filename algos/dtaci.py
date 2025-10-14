import numpy as np

class DtACI:
    """
    Distribution-free Time-series Adaptive Conformal Inference (DtACI)
    Implementation from https://arxiv.org/abs/2208.08401
    """
    
    def __init__(self, gammas, alpha=0.1, sigma=1/1000, eta=2.72, alpha_init=None, 
                 eta_adapt=False, eta_lookback=500):
        """
        Initialize DtACI algorithm
        
        Parameters:
        gammas : list or array
            Candidate gamma values for learning rates
        alpha : float, default=0.1
            Target miscoverage rate
        sigma : float, default=1/1000
            Mixing parameter for expert weights
        eta : float, default=2.72
            Learning rate parameter
        alpha_init : float, optional
            Initial alpha value (defaults to alpha)
        eta_adapt : bool, default=False
            Whether to adapt eta parameter
        eta_lookback : int, default=500
            Lookback window for adaptive eta
        """
        self.gammas = np.array(gammas)
        self.alpha = alpha
        self.sigma = sigma
        self.eta = eta
        self.alpha_init = alpha_init if alpha_init is not None else alpha
        self.eta_adapt = eta_adapt
        self.eta_lookback = eta_lookback
        self.k = len(gammas)
        
    def pinball_loss(self, u, alpha):
        """Pinball loss function"""
        return alpha * u - np.minimum(u, 0)
    
    def fit(self, betas):
        """
        Run DtACI algorithm on sequence of conformity scores
        
        Parameters:
        betas : array-like
            Sequence of conformity scores
            
        Returns:
        tuple: (alpha_seq, err_seq_adapt, err_seq_fixed, gamma_seq, 
                mean_alpha_seq, mean_err_seq, mean_gammas)
        """
        betas = np.array(betas)
        T = len(betas)
        
        # Initialize sequences
        alpha_seq = np.full(T, self.alpha_init)
        err_seq_adapt = np.zeros(T)
        err_seq_fixed = np.zeros(T)
        gamma_seq = np.zeros(T)
        mean_alpha_seq = np.zeros(T)
        mean_err_seq = np.zeros(T)
        mean_gammas = np.zeros(T)
        loss_seq = np.zeros(T)
        
        # Initialize expert parameters
        expert_alphas = np.full(self.k, self.alpha_init)
        expert_ws = np.ones(self.k)
        cur_expert = np.random.randint(0, self.k)
        expert_cumulative_losses = np.zeros(self.k)
        expert_probs = np.full(self.k, 1.0 / self.k)
        
        eta = self.eta
        
        for t in range(T):
            # Adapt eta if requested
            if t > self.eta_lookback and self.eta_adapt:
                lookback_losses = loss_seq[t-self.eta_lookback:t]
                eta = np.sqrt((np.log(2*self.k*self.eta_lookback) + 1) / 
                             np.sum(lookback_losses**2))
            
            # Current predictions
            alpha_t = expert_alphas[cur_expert]
            alpha_seq[t] = alpha_t
            err_seq_adapt[t] = float(alpha_t > betas[t])
            err_seq_fixed[t] = float(self.alpha > betas[t])
            gamma_seq[t] = self.gammas[cur_expert]
            mean_alpha_seq[t] = np.sum(expert_probs * expert_alphas)
            mean_err_seq[t] = float(mean_alpha_seq[t] > betas[t])
            mean_gammas[t] = np.sum(expert_probs * self.gammas)
            
            # Compute expert losses
            expert_losses = self.pinball_loss(betas[t] - expert_alphas, self.alpha)
            loss_seq[t] = np.sum(expert_losses * expert_probs)
            
            # Update expert alphas
            expert_alphas += self.gammas * (self.alpha - (expert_alphas > betas[t]).astype(float))
            
            # Update expert weights
            if eta < np.inf:
                # Use log-sum-exp trick for numerical stability
                log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
                # Subtract max for numerical stability
                log_weights_stable = log_weights - np.max(log_weights)
                expert_bar_ws = np.exp(log_weights_stable)

                # Normalize with epsilon to prevent division by zero
                expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
                expert_next_ws = ((1 - self.sigma) * expert_bar_ws / expert_bar_ws_sum +
                                self.sigma / self.k)

                # Normalize probabilities
                expert_probs = expert_next_ws / (np.sum(expert_next_ws) + 1e-300)
                # Ensure probabilities sum to 1 and are valid
                expert_probs = np.clip(expert_probs, 1e-300, 1.0)
                expert_probs = expert_probs / np.sum(expert_probs)

                cur_expert = np.random.choice(self.k, p=expert_probs)
                expert_ws = expert_next_ws
            else:
                expert_cumulative_losses += expert_losses
                cur_expert = np.argmin(expert_cumulative_losses)
                
        return (alpha_seq, err_seq_adapt, err_seq_fixed, gamma_seq,
                mean_alpha_seq, mean_err_seq, mean_gammas)


def dtaci(betas, alpha=0.1, gammas=[0.001, 0.01, 0.1], sigma=1/1000, eta=2.72, 
          alpha_init=None, eta_adapt=False, eta_lookback=500):
    """
    Convenience function to run DtACI algorithm
    
    Parameters:
    betas : array-like
        Sequence of conformity scores
    alpha : float, default=0.1
        Target miscoverage rate
    gammas : list, default=[0.001, 0.01, 0.1]
        Candidate gamma values
    sigma : float, default=1/1000
        Mixing parameter
    eta : float, default=2.72
        Learning rate parameter
    alpha_init : float, optional
        Initial alpha value
    eta_adapt : bool, default=False
        Whether to adapt eta
    eta_lookback : int, default=500
        Lookback window for adaptive eta
        
    Returns:
    tuple: (alpha_seq, err_seq_adapt, err_seq_fixed, gamma_seq,
            mean_alpha_seq, mean_err_seq, mean_gammas)
    """
    dtaci_instance = DtACI(gammas, alpha, sigma, eta, alpha_init, eta_adapt, eta_lookback)
    return dtaci_instance.fit(betas)