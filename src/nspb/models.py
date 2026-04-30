import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

# Bayesian Linear Regression Model
class BayesianModel():
  def __init__(self, d, default_variance=50, default_noise_precision=0.33):
    # d = amount of topics, or dimensions
    self.d = d

    # B = noise precision parameter, a known constant
    self.B = default_noise_precision

    # Setting all preferences to zero, and setting default covariance
    self.p = np.zeros(d)
    self.covariance = np.diag([default_variance] * d).astype(np.float64)
  
  def update(self, x, r):
    assert x.shape[0] == self.d
    x = x.reshape(1, -1)

    # Equations from Bishop, 2006
    orig_inv_covariance = np.linalg.inv(self.covariance)
    new_inv_covariance = orig_inv_covariance + self.B * (x.T @ x)
    new_covariance = np.linalg.inv(new_inv_covariance)

    self.p = (new_covariance @ ((orig_inv_covariance @ self.p) + (self.B * (x.T * r)).T).T).reshape(self.d)
    self.covariance = new_covariance

    return self.p, self.covariance
  
  def predict(self, x):
    return self.p @ x

  def get_params(self):
    return self.p, self.covariance

  def set_params(self, p, c):
    self.p = p
    self.covariance = c


# Variance Bounded Bayesian Linear Regression Model
class VarianceBoundedBayesianModel():
  def __init__(self, d, default_variance=15, default_noise_precision=0.33, tau=1.0):
    # d = amount of topics, or dimensions
    self.d = d

    # tau = the minimum value of the variance for each variable
    self.tau = tau

    # B = noise precision parameter, a known constant
    self.B = default_noise_precision

    # Setting all preferences to zero, and setting default covariance
    self.default_variance = default_variance
    self.p = np.zeros(d)
    self.covariance = np.diag([default_variance] * d).astype(np.float64)
  
  def update(self, x, r):
    assert x.shape[0] == self.d
    x = x.reshape(1, -1)
    
    e, v = scipy.linalg.eigh(self.covariance)                
    e_prime = np.clip(e, a_min=self.tau, a_max=None)
    s = v @ np.diag(e_prime) @ v.T
    s_inv = np.linalg.inv(s)

    new_inv_covariance = s_inv + self.B * (x.T @ x)
    new_covariance = np.linalg.inv(new_inv_covariance)

    self.p = (new_covariance @ ((s_inv @ self.p) + (self.B * (x.T * r)).T).T).reshape(self.d)
    self.covariance = new_covariance

    return self.p, s, self.covariance
  
  def predict(self, x):
    return self.p @ x

  def get_params(self):
    return self.p, self.covariance

  def set_params(self, p, c):
    self.p = p
    self.covariance = c
    

# Bayesian Linear Regression with Discounting / Forgetting Factor (From Dr. Gao's document -- Section 2.1.1)
class BayesianForgettingFactorModel():
    def __init__(self, d, default_variance=50.0, default_noise_precision=0.33):
        """
        Bayesian Linear Regression with a forgetting factor.
        """
        self.d = d # d = amount of topics, or dimensions
        self.B = default_noise_precision  # B = noise precision parameter, a known constant
        
        self.p = np.zeros(d)              # previous mean
        self.covariance = np.diag([default_variance] * d).astype(np.float64)  # previous covariance

    def update(self, x, r, rho=1.0):
        """
        One-sample update with forgetting factor.
        """
        assert 0.0 < rho <= 1.0
        assert x.shape[0] == self.d
        x = x.reshape(1, -1)

        # Previous precision matrix
        inv_prev = np.linalg.inv(self.covariance) 

        # New precision
        new_inv_cov = rho * inv_prev + self.B * (x.T @ x)

        # New covariance Σ_t
        new_cov = np.linalg.inv(new_inv_cov)

        # New mean μ_t
        rhs = rho * (inv_prev @ self.p) + self.B * (x.flatten() * r)
        new_mean = new_cov @ rhs

        self.p, self.covariance = new_mean, new_cov
        return self.p, self.covariance

    def predict(self, x):
        return float(self.p @ x)

    def get_params(self):
        return self.p, self.covariance

    def set_params(self, p, c):
        self.p, self.covariance = p, c



# Bayesian Linear Regression with Sliding Window
class BayesianSlidingWindowModel():
  def __init__(self, d, m=20, default_variance=50, default_noise_precision=0.33):
    # d = amount of topics, or dimensions
    self.d = d
    self.m = m

    # B = noise precision parameter, a known constant
    self.B = default_noise_precision

    # Setting all preferences to zero, and setting default covariance
    self.p = np.zeros(d)
    self.covariance = np.diag([default_variance] * d).astype(np.float64)

    # Save the initial prior (m0, S0) to reuse every update
    self._initial_p = self.p.copy()
    self._initial_covariance = self.covariance.copy()

    # Buffers for last m observations
    self.X_window = []
    self.R_window = []

  def update(self, x, r):
    assert x.shape[0] == self.d
    x = x.reshape(1, -1)

    # Maintain sliding window
    self.X_window.append(x)
    self.R_window.append(float(r))
    if len(self.X_window) > self.m:
      self.X_window.pop(0)
      self.R_window.pop(0)

    # Stack window
    X = np.vstack(self.X_window)                               # (k, d)
    R = np.asarray(self.R_window, dtype=float).reshape(-1, 1)  # (k, 1)

    # Equations from Bishop, 2006 (batch over window)
    # Use initial (m0, S0) every time
    orig_inv_covariance = np.linalg.inv(self._initial_covariance)          # S0^{-1}
    new_inv_covariance = orig_inv_covariance + self.B * (X.T @ X)
    new_covariance = np.linalg.inv(new_inv_covariance)

    # rhs = S0^{-1} m0 + β X^T R ; keep shapes explicit
    rhs = (orig_inv_covariance @ self._initial_p.reshape(-1, 1)) + (self.B * (X.T @ R))

    # p_new = Σ_new * rhs
    self.p = (new_covariance @ rhs).reshape(self.d)
    self.covariance = new_covariance

    return self.p, self.covariance

  def predict(self, x):
    return self.p @ x

  def get_params(self):
    return self.p, self.covariance

  def set_params(self, p, c):
    self.p = p
    self.covariance = c






# Adaptive Regularization of Weights for Regression Model
class AROW_Regression():
  def __init__(self, k_topics, lam1, lam2):
    self.k = k_topics
    self.mu = np.zeros([self.k, 1])

    self.cov = np.diag(np.ones(self.k))

    self.lam1 = lam1
    self.lam2 = lam2

  def update(self, x, y):
    x = np.reshape(x, (-1, 1))

    r1 = 1.0 / (2 * self.lam1)
    r2 = 1.0 / (2 * self.lam2)

    beta_mu =  1.0 / (x.T @ self.cov @ x + r1)
    alpha = (y - x.T @ self.mu) * beta_mu
    self.mu = self.mu + (alpha * (self.cov @ x))

    beta_sigma =  1.0 / (x.T @ self.cov @ x + r2)
    self.cov = self.cov - ((beta_sigma * self.cov) @ x @ x.T @ self.cov)

    return self.mu, self.cov

  def get_params(self):
    return self.mu, self.cov

  def predict(self, x):
    return self.mu.T @ x.reshape(-1, 1)
    

# Kalman Filter from Google Deepmind Paper    
class KalmanFilter():
  def __init__(self, d, variance_p=0.01, variance=0.05, delta = 0.0, eta= 0.001):
    # d = amount of topics, or dimensions
    self.d = d

    self.delta = delta
    self.eta = eta
    self.variance_p = variance_p
    self.variance = variance
    self.p = np.zeros(d)
    self.cov = np.diag([variance_p] * d).astype(np.float64)

  def update(self, x, r):
    #gamma = 0.998
    #prediction steps
    gamma = np.exp(-self.delta * 0.5)
    #print('gamma: ', gamma)
    p_overline = gamma * self.p
    #cov_overline = self.cov * gamma**2 + (1 - gamma**2) * self.variance_p
    cov_overline = (self.cov * gamma**2) + (1 - gamma**2) * self.variance_p * np.eye(self.d)


    #SGD update
    a = ((x.T @ self.cov @ x) - (x.T @ x) * self.variance_p) * np.exp(-self.delta)
    b = (x.T @ x) * self.variance_p + self.variance
    c = r - x @ self.p * np.exp(-self.delta * 0.5)
    d = x @ self.p * np.exp(-self.delta * 0.5)
    grad = ((a + b) * (a - c * d) - a * c**2) / (2 * (a + b)**2)
    self.delta = self.delta + self.eta * grad
    #delt = self.delta
    #self.delta = 0.01

    if self.delta < 0:
      self.delta = 0.0

    #update steps
    self.p = p_overline + (((cov_overline @ x) * (r - x @ p_overline)) / ((x.T @ cov_overline @ x) + self.variance))
    self.cov = cov_overline - ((cov_overline @ np.outer(x, x) @ cov_overline) / ((x.T @ cov_overline @ x) + self.variance))

    return self.p, self.cov

  def predict(self, x):
    return self.p @ x

  def get_params(self):
    return self.p, self.cov
    

# Normalized Least Mean Square Model
class NLMS():
  def __init__(self, k_topics, step_size = 0.1, eps = 0.001):
    # k_topics = amount of topics
    self.k = k_topics
    self.mean = np.zeros(k_topics)
    self.step_size = step_size
    self.eps = eps

  def update(self, x, y):
    e = y - self.predict(x)
    self.mean = self.mean +  (self.step_size * e * x) / (self.eps + x.T @ x)  
    return self.mean

  def predict(self, x):
    return self.mean @ x

  def get_params(self):
    return self.mean



# Power Prior Bayesian Linear Regression Model
class PowerPriorBayesianModel():
  def __init__(self, d, alpha=0.5, default_variance=50, default_noise_precision=0.33):
    # d = amount of topics, or dimensions
    self.d = d
    # alpha = power prior parameter α in (0,1]
    self.alpha = alpha
    # B = noise precision β (or 1/σ²). This is the Σ⁻¹ from the paper's math.
    self.B = default_noise_precision

    # Current posterior (will hold the step-τ posterior: μ_τ^(pp), Σ_τ^(pp))
    self.p = np.zeros(d)
    self.covariance = np.diag([default_variance] * d).astype(np.float64)

    # Fixed initial prior (m_0, S_0)
    self._m0 = self.p.copy()
    self._S0 = self.covariance.copy()
    # S0_inv = S_0⁻¹
    self._S0_inv = np.linalg.inv(self._S0)

    # --- Discounted historical sufficient statistics ---
    # These variables accumulate the α-discounted history.
    
    # _A accumulates the historical data precision: α * Φ_{τ-1}ᵀ * Σ⁻¹ * Φ_{τ-1}
    # This is equivalent to: αβ * Σ_{i=1}^{τ-1} x_i x_i^T
    self._A = np.zeros((d, d), dtype=np.float64)
    
    # _b accumulates the historical data mean-component: α * Φ_{τ-1}ᵀ * Σ⁻¹ * t_{τ-1}
    # This is equivalent to: αβ * Σ_{i=1}^{τ-1} x_i r_i
    self._b = np.zeros((d, 1), dtype=np.float64)

  def update(self, x, r):
    # x = current item vector (ϕ_τ), r = current rating (t_τ)
    assert x.shape[0] == self.d
    x = x.reshape(1, -1)      # (1, d), treated as ϕ_τᵀ
    r = float(r)              # t_τ

    # --- 1. Build the Power Prior for step τ (based on history up to τ-1) ---
    
    # Build power-prior precision:
    # Eq: Σ_{τ-1}^{-1,(pp)} = S_0⁻¹ + α * Φ_{τ-1}ᵀ * Σ⁻¹ * Φ_{τ-1}
    # Code: Sigma_prior_inv = S0_inv + _A
    Sigma_prior_inv = self._S0_inv + self._A

    # Build the right-hand-side (rhs) for the mean calculation:
    # Eq: S_0⁻¹ * m_0 + α * Φ_{τ-1}ᵀ * Σ⁻¹ * t_{τ-1}
    # Code: rhs_prior = (S0_inv @ m0) + _b
    # This term `rhs_prior` is mathematically equivalent to: Σ_{τ-1}^{-1,(pp)} * μ_{τ-1}^{(pp)}
    rhs_prior = (self._S0_inv @ self._m0.reshape(-1, 1)) + self._b

    # --- 2. Perform Posterior Update at Step τ (using new data x, r) ---
    # This section implements the "Posterior Update at Step τ" equations.

    # Calculate posterior precision for step τ:
    # Eq: Σ_τ^{-1,(pp)} = Σ_{τ-1}^{-1,(pp)} + ϕ_τ * Σ⁻¹ * ϕ_τᵀ
    # Code: new_inv_covariance = Sigma_prior_inv + B * (x.T @ x)
    new_inv_covariance = Sigma_prior_inv + self.B * (x.T @ x)
    # This is the final posterior covariance: Σ_τ^(pp)
    new_covariance = np.linalg.inv(new_inv_covariance) 

    # Calculate the full right-hand-side for the posterior mean:
    # Eq: ( Σ_{τ-1}^{-1,(pp)} * μ_{τ-1}^{(pp)} ) + ( ϕ_τ * Σ⁻¹ * t_τ )
    # Code: rhs = rhs_prior + (B * (x.T * r))
    rhs = rhs_prior + (self.B * (x.T * r))

    # Calculate the posterior mean:
    # Eq: μ_τ^{(pp)} = Σ_τ^{(pp)} * ( rhs )
    # Code: self.p = new_covariance @ rhs
    self.p = (new_covariance @ rhs).reshape(self.d)
    self.covariance = new_covariance

    # --- 3. Update Historical Stats for *Next* Step (τ+1) ---

    # Fold the current data (x, r) from step τ into the discounted history.
    # This data point, (ϕ_τ, t_τ), is now part of the history for the *next* update.
    
    # _A_new = _A_old + αβ * (ϕ_τᵀ * ϕ_τ)
    self._A += self.alpha * self.B * (x.T @ x)
    # _b_new = _b_old + αβ * (ϕ_τᵀ * t_τ)
    self._b += self.alpha * self.B * (x.T * r)

    return self.p, self.covariance

  def predict(self, x):
    return self.p @ x

  def get_params(self):
    return self.p, self.covariance

  def set_params(self, p, c):
    self.p = p
    self.covariance = c


class NormalInverseGammaModel():
    def __init__(self, d, default_variance=50.0, default_a=2.0, default_b=2.0):
        self.d = d
        self.p = np.zeros(d)
        self.V = np.diag([default_variance] * d).astype(np.float64)
        self.invV = np.diag([1.0/default_variance] * d).astype(np.float64)  # cache
        self.a = float(default_a)
        self.b = float(default_b)

    def update(self, x, r):
        assert x.shape[0] == self.d
        x = x.reshape(-1, 1)

        invV_new = self.invV + (x @ x.T)
        V_new = np.linalg.inv(invV_new)       # only 1 inverse now

        p_new = V_new @ (self.invV @ self.p.reshape(-1, 1) + x * r)
        p_new = p_new.flatten()

        a_new = self.a + 0.5

        quad_prev = float(self.p @ self.invV @ self.p)
        quad_new = float(p_new @ invV_new @ p_new)
        b_new = self.b + 0.5 * (float(r) ** 2 + quad_prev - quad_new)

        self.p, self.V, self.invV = p_new, V_new, invV_new
        self.a, self.b = float(a_new), float(b_new)
        return self.p, self.V, self.a, self.b

    def predict(self, x):
        return float(self.p @ x)

    def get_params(self):
        return self.p, self.V, self.a, self.b

    def set_params(self, p, V, a, b):
        self.p, self.V, self.a, self.b = p, V, a, b
        self.invV = np.linalg.inv(V)



class NormalWishartBayesianModel:
    def __init__(self, d, m0=None, kappa0=1.0, nu0=None, S0=None):
        self.d = int(d)

        # Initialize m as a proper column vector (d, 1)
        if m0 is None:
            self.m = np.zeros((self.d, 1), dtype=np.float64)
        else:
            self.m = np.asarray(m0, dtype=np.float64).reshape(self.d, 1)

        self.kappa = float(kappa0)

        # Degrees of freedom nu
        if nu0 is None:
            # nu0 > d - 1 is required for valid Inverse Wishart expectation
            nu0 = self.d + 2
        if nu0 <= self.d - 1:
            raise ValueError(f"nu0 must be > d - 1 (which is {self.d - 1}), but got {nu0}")
        self.nu = float(nu0)

        # Scale matrix S (d, d)
        if S0 is None:
            S0 = np.eye(self.d, dtype=np.float64)
        self.S = np.asarray(S0, dtype=np.float64).reshape(self.d, self.d)

    def update(self, x, t):
        """
        NIW-like update for scalar regression target.

        x : array-like, shape (d,) or (d,1)
            Feature/topic vector.
        t : float
            Scalar target/rating.

        Updates:
          kappa <- kappa + 1
          nu    <- nu + 1
          m     <- (kappa_prev*m_prev + x*t) / kappa
          S     <- S + (resid^2) * (x x^T) + (kappa_prev/kappa) * (m_prev - m)(m_prev - m)^T

        The residual term is injected as a PSD matrix in the direction of x
        to keep S as (d,d).
        """

        d = self.d
        x = np.asarray(x, dtype=np.float64).reshape(d, 1)
        t_val = float(t)

        # ---- cache previous params ----
        m_prev = self.m.copy()          # (d,1)
        kappa_prev = float(self.kappa)

        # ---- update kappa, nu ----
        self.kappa = kappa_prev + 1.0
        self.nu = float(self.nu) + 1.0

        # ---- update mean m ----
        # m = (kappa_prev*m_prev + x*t) / kappa
        self.m = (kappa_prev * m_prev + x * t_val) / self.kappa

        # ---- compute residual using updated mean ----
        pred = float(x.T @ self.m)      # scalar
        resid = t_val - pred            # scalar

        # ---- drift term (PSD) ----
        dm = (m_prev - self.m)          # (d,1)
        term_drift = (kappa_prev / self.kappa) * (dm @ dm.T)   # (d,d)

        # ---- residual term as PSD matrix (rank-1) ----
        # Map scalar residual energy into covariance direction of x
        term_resid = (resid ** 2) * (x @ x.T)                  # (d,d)

        # ---- update S ----
        self.S = self.S + term_drift + term_resid

        # ---- numerical stabilization: symmetrize + jitter ----
        self.S = 0.5 * (self.S + self.S.T)
        self.S = self.S + 1e-9 * np.eye(d)

        return self.kappa, self.m, self.nu, self.S


    def predict(self, x):
        """
        Returns the mean prediction.
        """
        x = np.asarray(x, dtype=np.float64).reshape(self.d, 1)
        # m is (d, 1), x is (d, 1). We want scalar output.
        # Use m.T @ x
        return float(self.m.T @ x)

    def get_params(self):
        return self.kappa, self.m, self.nu, self.S

    def set_params(self, kappa, m, nu, S):
        self.kappa = kappa
        self.m = np.asarray(m).reshape(self.d, 1)
        self.nu = nu
        self.S = np.asarray(S).reshape(self.d, self.d)