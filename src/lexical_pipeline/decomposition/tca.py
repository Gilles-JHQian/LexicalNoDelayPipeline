"""Tensor Component Analysis (CP/PARAFAC) math (dataset-agnostic).

Moved verbatim from ``tca_decomposition.py``.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac


def run_tca(tensor, rank, n_iter_max=500, random_state=42, normalize_factors=True):
    """Run CP/PARAFAC decomposition on a 3-way tensor.

    Parameters
    ----------
    tensor : np.ndarray, shape (n_electrodes, n_times, 2)
    rank : int
        Number of TCA components.
    normalize_factors : bool
        If True, normalize factor columns to unit norm (weights absorbed
        into the `weights` array).

    Returns
    -------
    weights : np.ndarray, shape (rank,)
        Component weights (norms of factor columns when normalize_factors=True).
    factors : list of np.ndarray
        [A (n_electrodes, rank), B (n_times, rank), C (2, rank)]
    reconstruction_error : float
        Relative Frobenius reconstruction error.
    """
    tl.set_backend('numpy')
    # Ensure float32 to halve PARAFAC working memory (khatri-rao, unfoldings).
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32, copy=False)
    T = tl.tensor(tensor)

    # NOTE: init="svd" is unsafe here. The lexicality mode has size 2, so the
    # mode-2 unfolding is shape (2, N*T). tensorly's truncated_svd sets
    # full_matrices=True whenever rank > min(shape)=2, and np.linalg.svd then
    # materializes Vh of shape (N*T, N*T) — e.g. 100+ GB for N=721, T=161.
    # Random init avoids this and is the standard choice when one mode is tiny.
    # return_errors=True gives us the per-iter normalized reconstruction error
    # (||T - T_hat||_F / ||T||_F), so we skip materializing T_recon entirely.
    cp, errors = parafac(
        T,
        rank=rank,
        n_iter_max=n_iter_max,
        init="random",
        random_state=random_state,
        normalize_factors=normalize_factors,
        return_errors=True,
    )

    weights = np.asarray(cp.weights)                     # (rank,)
    factors = [np.asarray(f) for f in cp.factors]        # list of 3
    recon_err = float(errors[-1])

    return weights, factors, recon_err
