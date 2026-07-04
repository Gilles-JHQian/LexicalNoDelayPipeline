"""High-gamma-activity statistics helpers (dataset-agnostic).

Reusable machinery shared by the per-dataset HGA drivers (time_perm_bands and the
diff_* contrast scripts): trial-level outlier removal, and (to come) the
window-restricted permutation-cluster + embed/grow-extent routine that those
scripts currently duplicate inline.
"""
import logging

import numpy as np


def trial_level_outlier_removal(epochs, copy=False):
    """Remove trials with globally elevated power using IQR method.

    Identifies trials whose mean power across all channels and time points
    exceeds Q3 + 1.5 * IQR, and sets the entire trial to NaN. This catches
    global artifacts (e.g. EMG, movement) that elevate high-gamma power
    uniformly across all channels but are too moderate for per-channel
    outlier detection (outliers_to_nan) to flag.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs to clean. Modified in-place unless copy=True.
    copy : bool
        If True, operate on a copy.

    Returns
    -------
    mne.BaseEpochs
        The cleaned epochs (same object if copy=False).
    """
    if copy:
        epochs = epochs.copy()
    data = epochs.get_data(copy=False)
    # Mean power per trial across all channels and time points
    trial_means = np.nanmean(data, axis=(1, 2))
    q1 = np.percentile(trial_means, 25)
    q3 = np.percentile(trial_means, 75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    outlier_mask = trial_means > upper_fence
    n_outlier = outlier_mask.sum()
    if n_outlier > 0:
        logging.info(f"  Trial-level IQR: removing {n_outlier}/{len(epochs)} "
                     f"trials (global mean > {upper_fence:.4g})")
        data[outlier_mask] = np.nan
    return epochs
