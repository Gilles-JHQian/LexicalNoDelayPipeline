"""Semi-NMF factorization math (dataset-agnostic).

Semi-NMF (Ding, Li & Jordan, 2010): V ~= W H with W >= 0 and H real-valued, so H
can represent below-baseline deflections without the negative-handling step that
classical NMF needs. Moved verbatim from ``semi_nmf_decomposition_concat_all.py``.
"""
import logging

import numpy as np
from sklearn.cluster import KMeans

# Re-export: the original driver defined its own (identical-behavior) copy.
from .nmf import compute_reconstruction_error  # noqa: F401

logger = logging.getLogger(__name__)


def _split_pos_neg(M):
    """Return (M_pos, M_neg) with M = M_pos - M_neg, both element-wise >= 0."""
    M_pos = np.maximum(M, 0.0)
    M_neg = np.maximum(-M, 0.0)
    return M_pos, M_neg


def _init_W_kmeans(V, k, random_state=42, eps=0.2):
    """K-means initialization of the non-negative factor W.

    Following Ding et al. (2010): cluster the rows of V (electrodes) into
    k groups, then set W[i, j] = 1 if electrode i is in cluster j else 0.
    Add a small constant to keep entries strictly positive (the
    multiplicative update divides by W and would otherwise lock zeros).
    """
    n, _ = V.shape
    V_clean = np.nan_to_num(V, nan=0.0)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(V_clean)
    W = np.zeros((n, k), dtype=np.float64)
    W[np.arange(n), labels] = 1.0
    W += eps
    return W


def _solve_H(V, W, ridge=1e-8):
    """Closed-form least-squares solve for H given W (mixed sign allowed).

    H = (Wᵀ W + ridge I)⁻¹ Wᵀ V   (small ridge for numerical safety).
    """
    k = W.shape[1]
    WtW = W.T @ W + ridge * np.eye(k, dtype=W.dtype)
    return np.linalg.solve(WtW, W.T @ V)


def run_semi_nmf(V, k, max_iter=500, tol=1e-5, random_state=42, eps=1e-12,
                 verbose=True):
    """Semi-NMF: V ≈ W H with W ≥ 0 and H real-valued.

    Parameters
    ----------
    V : np.ndarray, shape (n, m)
        Mixed-sign data matrix. NaNs are replaced with 0.
    k : int
        Number of components.
    max_iter : int
        Maximum number of alternating-update iterations.
    tol : float
        Relative change in Frobenius reconstruction error below which
        iteration stops.
    random_state : int
        Seed for the k-means initialization.
    eps : float
        Numerical floor used in divisions and square roots.

    Returns
    -------
    W : np.ndarray, shape (n, k)         non-negative
    H : np.ndarray, shape (k, m)         mixed sign
    info : dict   with keys ``n_iter``, ``errors``, ``converged``.
    """
    V = np.nan_to_num(V, nan=0.0).astype(np.float64, copy=False)
    n, m = V.shape

    W = _init_W_kmeans(V, k, random_state=random_state)
    H = _solve_H(V, W)

    V_norm = np.linalg.norm(V, 'fro')
    prev_err = np.linalg.norm(V - W @ H, 'fro') / max(V_norm, eps)
    errors = [prev_err]
    converged = False

    for it in range(1, max_iter + 1):
        # --- H update: closed-form least squares (unconstrained) ---
        H = _solve_H(V, W)

        # --- W update: multiplicative, keeps W >= 0 ---
        VHt = V @ H.T            # (n, k)
        HHt = H @ H.T            # (k, k)
        VHt_pos, VHt_neg = _split_pos_neg(VHt)
        HHt_pos, HHt_neg = _split_pos_neg(HHt)

        numer = VHt_pos + W @ HHt_neg
        denom = VHt_neg + W @ HHt_pos
        W = W * np.sqrt((numer + eps) / (denom + eps))

        # Guard: drop any NaNs/Infs that could arise from degenerate cols.
        if not np.all(np.isfinite(W)):
            W = np.nan_to_num(W, nan=eps, posinf=eps, neginf=eps)

        err = np.linalg.norm(V - W @ H, 'fro') / max(V_norm, eps)
        errors.append(err)

        if verbose and (it % 50 == 0 or it == 1):
            logger.info(f"  semi-NMF iter {it:4d}: rel err = {err:.6f}")

        if abs(prev_err - err) / max(prev_err, eps) < tol:
            converged = True
            if verbose:
                logger.info(
                    f"  Converged at iter {it} (rel err = {err:.6f})"
                )
            break
        prev_err = err

    # Final H solve so H is consistent with the final W.
    H = _solve_H(V, W)
    final_err = np.linalg.norm(V - W @ H, 'fro') / max(V_norm, eps)
    errors.append(final_err)

    return W, H, {
        "n_iter": len(errors) - 1,
        "errors": np.asarray(errors, dtype=np.float64),
        "converged": converged,
    }
