import numpy as np

class AgACI:
    """
    Aggregated Adaptive Conformal Inference (AgACI) method
    Implementation of the algorithm from the R code in AgACI.R
    """
    
    def __init__(self, gammas, alpha=0.1, alpha_init=None, eps=0.001):
        """
        Initialize AgACI algorithm
        
        Parameters:
        gammas : list or array
            Candidate gamma values for learning rates
        alpha : float, default=0.1
            Target miscoverage rate
        alpha_init : float, optional
            Initial alpha value (defaults to alpha)
        eps : float, default=0.001
            Small epsilon value for numerical stability
        """
        self.gammas = np.array(gammas)
        self.alpha = alpha
        self.alpha_init = alpha_init if alpha_init is not None else alpha
        self.eps = eps
        self.k = len(gammas)
        
    def pinball_loss(self, u, alpha):
        """Pinball loss function"""
        return alpha * u - np.minimum(u, 0)
    
    def fit(self, betas):
        """
        Run AgACI algorithm on sequence of conformity scores
        
        Parameters:
        betas : array-like
            Sequence of conformity scores
            
        Returns:
        tuple: (alpha_seq, err_seq, gamma_seq)
            alpha_seq: sequence of alpha values
            err_seq: sequence of errors
            gamma_seq: sequence of gamma values
        """
        betas = np.array(betas)
        T = len(betas)
        
        # Initialize sequences
        alpha_seq = np.full(T, self.alpha_init)
        err_seq = np.zeros(T)
        gamma_seq = np.zeros(T)
        
        # Initialize expert parameters
        expert_alphas = np.full(self.k, self.alpha_init)
        expert_probs = np.full(self.k, 1.0 / self.k)
        expert_sq_losses = np.zeros(self.k)
        expert_etas = np.zeros(self.k)
        expert_l_values = np.zeros(self.k)
        expert_max_losses = np.zeros(self.k)
        
        for t in range(T):
            # Compute predictions
            alpha_seq[t] = np.sum(expert_probs * expert_alphas)
            err_seq[t] = float(alpha_seq[t] > betas[t])
            gamma_seq[t] = np.sum(expert_probs * self.gammas)
            
            # Update expert weights
            expert_losses = (err_seq[t] - self.alpha) * (expert_alphas - alpha_seq[t])
            expert_sq_losses += expert_losses**2
            expert_max_losses = np.maximum(expert_max_losses, np.abs(expert_losses))
            
            expert_e_vals = 2**(np.ceil(np.log2(np.abs(expert_max_losses) + self.eps)) + 1)
            expert_l_values += 0.5 * (expert_losses * (1 + expert_etas * expert_losses) + 
                                    expert_e_vals * (expert_etas * expert_losses > 0.5))
            
            expert_etas = np.minimum(1.0 / expert_e_vals, 
                                   np.sqrt(np.log(self.k) / np.maximum(expert_sq_losses, self.eps)))
            
            # Update expert alphas
            expert_alphas += self.gammas * (self.alpha - (expert_alphas > betas[t]).astype(float))
            
            # Update expert weights and probabilities
            max_val = np.max(expert_etas * expert_l_values)
            expert_weights = expert_etas * np.exp(-expert_etas * expert_l_values + max_val)
            # Add epsilon to prevent division by zero
            expert_probs = expert_weights / (np.sum(expert_weights) + 1e-300)
            
        return alpha_seq, err_seq, gamma_seq


def agaci(betas, alpha=0.1, gammas=[0.001, 0.01, 0.1], alpha_init=None, eps=0.001):
    """
    Convenience function to run AgACI algorithm
    
    Parameters:
    betas : array-like
        Sequence of conformity scores
    alpha : float, default=0.1
        Target miscoverage rate
    gammas : list, default=[0.001, 0.01, 0.1]
        Candidate gamma values
    alpha_init : float, optional
        Initial alpha value
    eps : float, default=0.001
        Small epsilon for numerical stability
        
    Returns:
    tuple: (alpha_seq, err_seq, gamma_seq)
    """
    agaci_instance = AgACI(gammas, alpha, alpha_init, eps)
    return agaci_instance.fit(betas)