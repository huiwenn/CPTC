import numpy as np

class MVP:
    """
    Multi-Valid Prediction (MVP) algorithm
    Implementation of the algorithm from MVP.R
    """
    
    def __init__(self, alpha=0.1, m=40, r=800000, epsilon=1, K_eps=None):
        """
        Initialize MVP algorithm
        
        Parameters:
        alpha : float, default=0.1
            Target miscoverage rate
        m : int, default=40
            Number of discretization levels
        r : int, default=800000
            Regularization parameter
        epsilon : float, default=1
            Smoothness parameter
        K_eps : float, optional
            Precomputed K_epsilon value (computed if None)
        """
        self.alpha = alpha
        self.m = m
        self.r = r
        self.epsilon = epsilon
        self.K_eps = K_eps if K_eps is not None else self.compute_k_eps()
        self.eta = np.sqrt(np.log(m) / (2 * self.K_eps * m))
        
    def compute_k_eps(self, num_err=1e-7):
        """
        Compute K_epsilon parameter
        
        Parameters:
        num_err : float, default=1e-7
            Numerical error threshold
            
        Returns:
        float: K_epsilon value
        """
        total = 0
        i = 0
        while True:
            new_term = 1 / ((i + 1) * np.log(i + 2)**(1 + self.epsilon))
            total += new_term
            if new_term < num_err:
                return total
            i += 1
            
    def f(self, n):
        """Helper function for MVP algorithm"""
        return np.sqrt((n + 1) * np.log2(n + 2)**(1 + self.epsilon))
    
    def fit(self, scores):
        """
        Run MVP algorithm on sequence of scores
        
        Parameters:
        scores : array-like
            Sequence of conformity scores
            
        Returns:
        tuple: (alpha_seq, err_seq)
            alpha_seq: sequence of threshold values
            err_seq: sequence of errors
        """
        scores = np.array(scores)
        T = len(scores)
        
        # Initialize variables
        ns = np.zeros(self.m)
        Vs = np.zeros(self.m)
        Cs = np.zeros(self.m)
        err_seq = np.zeros(T)
        alpha_seq = np.zeros(T)
        
        for i in range(T):
            found_flag = False
            
            # Search for appropriate threshold
            for j in range(self.m - 1):
                if Vs[j] * Vs[j + 1] <= 0 and not found_flag:
                    found_flag = True
                    
                    # Compute probability p
                    if abs(Cs[j + 1]) + abs(Cs[j]) == 0:
                        p = 1
                    else:
                        p = abs(Cs[j + 1]) / (abs(Cs[j + 1]) + abs(Cs[j]))
                    
                    # Randomized selection
                    Z = np.random.binomial(1, p)
                    
                    if Z == 1:
                        threshold = j / self.m - 1 / (self.r * self.m)
                        err_seq[i] = float(scores[i] > threshold)
                        ns[j] += 1
                        Vs[j] += (1 - err_seq[i] - (1 - self.alpha))
                        
                        f_val = self.f(ns[j])
                        if f_val > 0:
                            Cs[j] = ((np.exp(self.eta * Vs[j] / f_val) - 
                                     np.exp(-self.eta * Vs[j] / f_val)) / f_val)
                    else:
                        threshold = (j + 1) / self.m
                        err_seq[i] = float(scores[i] > threshold)
                        ns[j + 1] += 1
                        Vs[j + 1] += (1 - err_seq[i] - (1 - self.alpha))
                        
                        f_val = self.f(ns[j + 1])
                        if f_val > 0:
                            Cs[j + 1] = ((np.exp(self.eta * Vs[j + 1] / f_val) - 
                                         np.exp(-self.eta * Vs[j + 1] / f_val)) / f_val)
                    
                    alpha_seq[i] = threshold
                    break
            
            # Handle edge cases when no threshold found
            if not found_flag:
                if Vs[0] < 0:
                    threshold = 1
                    err_seq[i] = 0
                    ns[self.m - 1] += 1
                    Vs[self.m - 1] += (1 - err_seq[i] - (1 - self.alpha))
                    
                    f_val = self.f(ns[self.m - 1])
                    if f_val > 0:
                        Cs[self.m - 1] = ((np.exp(self.eta * Vs[self.m - 1] / f_val) - 
                                          np.exp(-self.eta * Vs[self.m - 1] / f_val)) / f_val)
                else:
                    threshold = 0
                    err_seq[i] = 1
                    ns[0] += 1
                    Vs[0] += (1 - err_seq[i] - (1 - self.alpha))
                    
                    f_val = self.f(ns[0])
                    if f_val > 0:
                        Cs[0] = ((np.exp(self.eta * Vs[0] / f_val) - 
                                 np.exp(-self.eta * Vs[0] / f_val)) / f_val)
                
                alpha_seq[i] = threshold
                
        return alpha_seq, err_seq


def mvp(scores, alpha=0.1, m=40, r=800000, epsilon=1, K_eps=None):
    """
    Convenience function to run MVP algorithm
    
    Parameters:
    scores : array-like
        Sequence of conformity scores
    alpha : float, default=0.1
        Target miscoverage rate
    m : int, default=40
        Number of discretization levels
    r : int, default=800000
        Regularization parameter
    epsilon : float, default=1
        Smoothness parameter
    K_eps : float, optional
        Precomputed K_epsilon value
        
    Returns:
    tuple: (alpha_seq, err_seq)
    """
    mvp_instance = MVP(alpha, m, r, epsilon, K_eps)
    return mvp_instance.fit(scores)