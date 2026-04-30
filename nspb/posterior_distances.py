import numpy as np
import scipy
import scipy.linalg
from scipy.special import multigammaln, digamma, gammaln


def kl_divergence_niw(prev_k, prev_m, prev_v, prev_S, k, m, v, S):
    """
    Computes KL(NIW_tau || NIW_{tau-1}).
    
    Mapping:
    - (k, m, v, S)           -> parameters at step tau (current)
    - (prev_k, prev_m, prev_v, prev_S) -> parameters at step tau-1 (previous)
    """
    
    # Dimensions - derive from Covariance Matrix to be safe against (1, d) inputs for m
    d = S.shape[-1]
    
    # --- Aliases for formula matching (tau and tau-1) ---
    # _t   = tau
    # _tm1 = tau - 1
    k_t, m_t, v_t, S_t = k, m, v, S
    k_tm1, m_tm1, v_tm1, S_tm1 = prev_k, prev_m, prev_v, prev_S

    # Ensure means are 1D arrays to prevent shape mismatches (e.g. (1, d) vs (d, 1))
    m_t = np.array(m_t).reshape(-1)
    m_tm1 = np.array(m_tm1).reshape(-1)

    # --- Helpers ---
    # Multivariate Digamma Function
    # psi_d(x) = sum_{i=1}^d psi(x + (1-i)/2)
    def mv_digamma(x, d):
        res = 0
        for i in range(1, d + 1):
            res += digamma(x + (1 - i) / 2)
        return res

    # --- Pre-computations ---
    # Inverses
    inv_S_t = np.linalg.inv(S_t)
    inv_S_tm1 = np.linalg.inv(S_tm1)
    
    # Log Determinants (using slogdet for stability)
    _, logdet_S_t = np.linalg.slogdet(S_t)
    _, logdet_S_tm1 = np.linalg.slogdet(S_tm1)

    # Difference in means
    diff_m = m_t - m_tm1

    # --- Term 1 Calculation ---
    # 0.5 * [ d*k_t/k_tm1 + k_tm1*v_t*(diff_m^T S_t^-1 diff_m) - d + d*log(k_tm1/k_t) ]
    
    term1_quad = diff_m.T @ inv_S_t @ diff_m # (m_t - m_{t-1})^T S_t^{-1} (m_t - m_{t-1})
    
    term1 = (d * k_t / k_tm1) + \
            (k_tm1 * v_t * term1_quad) - \
            d + \
            (d * np.log(k_tm1 / k_t))

    # --- Term 2 Calculation ---
    # 0.5 * [ v_tm1(log|S_tm1| - log|S_t|) + v_t*tr(S_tm1^-1 S_t) + 2*log_gamma_ratio + ... ]
    
    trace_term = np.trace(inv_S_tm1 @ S_t) # tr(S_{t-1}^{-1} S_t)
    
    # Log Gamma Ratio: log( Gamma_d(v_tm1/2) / Gamma_d(v_t/2) )
    # Becomes: multigammaln(v_tm1/2) - multigammaln(v_t/2)
    log_gamma_diff = multigammaln(v_tm1 / 2, d) - multigammaln(v_t / 2, d)
    
    psi_term = (v_t - v_tm1) * mv_digamma(v_t / 2, d)
    
    term2 = v_tm1 * (logdet_S_tm1 - logdet_S_t) + \
            v_t * trace_term + \
            2 * log_gamma_diff + \
            psi_term - \
            (v_t * d)

    return 0.5 * (term1 + term2)

def kl_divergence_zero(cov_t, cov_tp):
  k = cov_t.shape[-1]
  inv_cov_tp = np.linalg.inv(cov_tp)

  # print("np.linalg.det(cov_tp1):", np.linalg.det(cov_tp1))
  # print("np.linalg.det(cov_t):", np.linalg.det(cov_t))
  # print("(np.linalg.det(cov_tp1) / np.linalg.det(cov_t)):", (np.linalg.det(cov_tp1) / np.linalg.det(cov_t)))
  out = np.trace((inv_cov_tp @ cov_t)) - k + np.log((np.linalg.det(cov_tp) / np.linalg.det(cov_t)))

  return (out / 2)

"""
def kl_divergence(cov_t, cov_tp1, mu_t, mu_tp1):
  k = cov_t.shape[-1]
  inv_cov_tp1 = np.linalg.inv(cov_tp1)

  out = np.trace((inv_cov_tp1 @ cov_t)) - k + (mu_t - mu_tp1).T @ inv_cov_tp1 @ (mu_t - mu_tp1) + \
        np.log((np.linalg.det(cov_tp1) / np.linalg.det(cov_t)))

  return (out / 2)
"""



def nig_kl_divergence(V_t, V_tp1, mu_t, mu_tp1, a_t, a_tp1, b_t, b_tp1, invV_t=None):
  k = V_t.shape[-1]
  if invV_t is None:
      invV_t = np.linalg.inv(V_t)

  sign_t, logdet_t = np.linalg.slogdet(V_t)
  sign_tp1, logdet_tp1 = np.linalg.slogdet(V_tp1)

  kl_cond = np.trace(invV_t @ V_tp1) - k + (logdet_t - logdet_tp1)
  e_inv_sigma2_q = a_tp1 / b_tp1
  dmu = (mu_t - mu_tp1).reshape(-1, 1)
  kl_cond = 0.5 * (kl_cond + float((dmu.T @ invV_t @ dmu)) * e_inv_sigma2_q)

  kl_ig = a_t * (np.log(b_tp1) - np.log(b_t)) + (a_tp1 - a_t) * digamma(a_tp1) + gammaln(a_t) - gammaln(a_tp1) + a_tp1 * (b_t / b_tp1 - 1.0)

  return float(kl_cond + kl_ig)


def kl_divergence(cov_t, cov_tp1, mu_t, mu_tp1):
  k = cov_t.shape[-1]
  inv_cov_tp1 = np.linalg.inv(cov_tp1)

  # Use slogdet instead of det to avoid overflow
  sign_t, logdet_t = np.linalg.slogdet(cov_t)
  sign_tp1, logdet_tp1 = np.linalg.slogdet(cov_tp1)

  out = np.trace((inv_cov_tp1 @ cov_t)) - k + (mu_t - mu_tp1).T @ inv_cov_tp1 @ (mu_t - mu_tp1) + \
        (logdet_tp1 - logdet_t)

  return (out / 2)


def wasserstein_distance(cov_t, cov_tp, mu_t, mu_tp, squared: bool = False, eps: float = 1e-9):
    cov_t = np.asarray(cov_t, dtype=np.float64)
    cov_tp = np.asarray(cov_tp, dtype=np.float64)
    mu_t = np.asarray(mu_t, dtype=np.float64).reshape(-1)
    mu_tp = np.asarray(mu_tp, dtype=np.float64).reshape(-1)

    d = cov_t.shape[-1]

    # Basic sanity checks (helps catch swapped args like your KL issue)
    if cov_t.ndim != 2 or cov_tp.ndim != 2:
        raise ValueError(f"covariances must be 2D; got {cov_t.shape} and {cov_tp.shape}")
    if cov_t.shape != (d, d) or cov_tp.shape != (d, d):
        raise ValueError(f"covariances must be square; got {cov_t.shape} and {cov_tp.shape}")
    if mu_t.shape[0] != d or mu_tp.shape[0] != d:
        raise ValueError(f"mean dim mismatch: mu_t {mu_t.shape}, mu_tp {mu_tp.shape}, cov dim {d}")

    cov_t = 0.5 * (cov_t + cov_t.T) + eps * np.eye(d)
    cov_tp = 0.5 * (cov_tp + cov_tp.T) + eps * np.eye(d)
    diff = mu_t - mu_tp
    mean_term = float(diff @ diff)  # ||mu_t - mu_tp||^2

    # Covariance term: Tr(S_t + S_tp1 - 2 * sqrtm( sqrt(S_tp1) S_t sqrt(S_tp1) ))
    sqrt_cov_tp = scipy.linalg.sqrtm(cov_tp)
    # sqrtm may return complex due to numerical noise; discard tiny imag parts
    if np.iscomplexobj(sqrt_cov_tp):
        sqrt_cov_tp = sqrt_cov_tp.real

    mid = sqrt_cov_tp @ cov_t @ sqrt_cov_tp
    sqrt_mid = scipy.linalg.sqrtm(mid)
    if np.iscomplexobj(sqrt_mid):
        sqrt_mid = sqrt_mid.real

    cov_term = float(np.trace(cov_t + cov_tp - 2.0 * sqrt_mid))

    # Numerical guard: sometimes cov_term can go slightly negative (~ -1e-12)
    w2_sq = mean_term + max(0.0, cov_term)

    return w2_sq if squared else float(np.sqrt(w2_sq))



def wasserstein_distance_niw(prev_k, prev_m, prev_v, prev_S,
                             k, m, v, S,
                             squared=False, eps=1e-9):
    # ---- helper: NIW -> Gaussian ----
    def niw_to_gaussian(kappa, m, nu, S):
        m = np.asarray(m).reshape(-1)
        d = m.shape[0]
        if nu <= d + 1:
            raise ValueError("Need nu > d + 1 for E[Sigma].")

        S = np.asarray(S, dtype=np.float64)
        E_Sigma = S / (nu - d - 1)
        cov_mu = E_Sigma / kappa
        return m, cov_mu

    # Previous Gaussian
    mu_prev, cov_prev = niw_to_gaussian(prev_k, prev_m, prev_v, prev_S)
    # Current Gaussian
    mu_curr, cov_curr = niw_to_gaussian(k, m, v, S)

    # ---- now reuse Gaussian Wasserstein ----
    return wasserstein_distance(
        cov_curr, cov_prev, mu_curr, mu_prev,
        squared=squared, eps=eps
    )
