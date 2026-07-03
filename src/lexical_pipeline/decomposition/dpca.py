"""demixed PCA (dPCA) math (dataset-agnostic).

Moved verbatim from ``dpca_decomposition.py``. The ``labels`` default ('cdt')
matches that script's ``DPCA_LABELS`` constant; callers pass their own label
string for a different factor structure.

Requires the ``dPCA`` package; imported lazily at module load so the rest of
``lexical_pipeline.decomposition`` does not depend on it.
"""
import numpy as np
from dPCA.dPCA import dPCA


def run_dpca(X, trialX, n_components, labels='cdt', regularizer=None):
    """Fit dPCA and return results.

    Parameters
    ----------
    X : np.ndarray, shape (N, n_cond, n_lex, T)
    trialX : np.ndarray, shape (N, n_cond, n_lex, max_trials, T)
    n_components : int
    labels : str
        dPCA label string, e.g. 'cdt'.
    regularizer : str, float, or None
        None for no regularization, 'auto' for cross-validated,
        or a numeric value for manual regularization.

    Returns
    -------
    dpca_model : dPCA instance (fitted)
    Z : dict  {marginalization_key: np.ndarray}
        Projected data for each marginalization.
    """
    dpca_model = dPCA(
        labels=labels,
        n_components=n_components,
        regularizer=regularizer,
    )
    # When using auto-regularization, protect the time axis so that
    # cross-validation splits trials but keeps time-points together.
    if regularizer == 'auto':
        dpca_model.protect = ['t']
    dpca_model.fit(X, trialX)
    Z = dpca_model.transform(X)  # dict of arrays
    return dpca_model, Z


def compute_explained_variance(dpca_model, X):
    """Compute total and per-marginalization explained variance."""
    Z = dpca_model.transform(X)
    # Total variance in the data
    X_flat = X.reshape(X.shape[0], -1)  # (N, cond*lex*T)
    total_var = np.sum((X_flat - X_flat.mean(axis=1, keepdims=True)) ** 2)

    explained = {}
    for key, z_marg in Z.items():
        # z_marg shape: (n_components, n_cond, n_lex, T)
        # Each component explains variance
        explained[key] = np.sum(z_marg ** 2)

    return total_var, explained
