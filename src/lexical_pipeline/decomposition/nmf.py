"""NMF factorization math (dataset-agnostic).

Canonical copies of the NMF helpers that had drifted (docstring/comment/source-
whitespace only) into several ``decomposition/*.py`` scripts. Numerical behavior is
identical to every one of those originals.
"""
import logging

import numpy as np
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


def normalize_channels(V):
    """Z-score normalize each channel across time."""
    ch_mean = np.nanmean(V, axis=1, keepdims=True)
    ch_std = np.nanstd(V, axis=1, keepdims=True)
    ch_std[ch_std < 1e-10] = 1.0
    V_norm = (V - ch_mean) / ch_std
    return V_norm, ch_mean.ravel(), ch_std.ravel()


def truncate_negatives(V, report=True):
    """Truncate negative values to 0 for standard NMF."""
    n_neg = np.sum(V < 0)
    n_total = V.size
    if report:
        logger.info(f"Truncating {n_neg} negative values "
                     f"({n_neg/n_total*100:.2f}% of {n_total} total)")
    return np.clip(V, 0, None), 0.0


def shift_to_nonneg(V, percentile=5, report=True):
    """Shift matrix so that the given percentile becomes 0.

    V_shifted = V - pct_value  (then clip residual negatives to 0)
    """
    clean = V[~np.isnan(V)]
    pct_val = np.percentile(clean, percentile)
    if report:
        logger.info(f"Shift-NMF: {percentile}th percentile = {pct_val:.6f}")
        logger.info(f"  Shifting entire matrix by -{pct_val:.6f}")
    V_shifted = V - pct_val
    n_still_neg = np.sum(V_shifted < 0)
    if report:
        logger.info(f"  Residual negatives after shift: {n_still_neg} "
                     f"({n_still_neg/V.size*100:.2f}%)")
    V_shifted = np.clip(V_shifted, 0, None)
    return V_shifted, pct_val


def run_nmf(V, k, max_iter=1000, random_state=42, init='nndsvda'):
    """Run NMF on matrix V.

    Parameters
    ----------
    V : np.ndarray, shape (n_samples, n_features)
        Non-negative input matrix.
    k : int
        Number of components.

    Returns
    -------
    W : np.ndarray, shape (n_samples, k)
    H : np.ndarray, shape (k, n_features)
    model : NMF instance
    """
    model = NMF(
        n_components=k,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        solver='mu',  # multiplicative update; handles zeros well
        beta_loss='frobenius',
    )
    W = model.fit_transform(V)
    H = model.components_
    return W, H, model


def compute_reconstruction_error(V, W, H):
    """Frobenius norm of (V - WH) / Frobenius norm of V."""
    residual = np.linalg.norm(V - W @ H, 'fro')
    total = np.linalg.norm(V, 'fro')
    return residual / total if total > 0 else np.inf
