"""High-gamma-activity statistics helpers (dataset-agnostic).

Reusable machinery shared by the per-dataset HGA drivers (time_perm_bands and the
diff_* contrast scripts): trial-level outlier removal, and the window-restricted
permutation-cluster + embed/grow-extent routine that those scripts used to
duplicate inline.
"""
import logging

import numpy as np
from ieeg.calc.stats import time_perm_cluster


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


def windowed_cluster_with_extent(
    win_act, win_bsl, full_act, full_bsl,
    n_channels, n_times, w0,
    p_thresh=0.1, n_perm=5000, tails=1, ignore_adjacency=1, n_jobs=-1,
):
    """Window-restricted permutation cluster test, embedded + grown to full extent.

    The significance decision is made by a permutation cluster test over the
    window only (``win_act`` vs ``win_bsl``), so the cluster-size null is built
    from exactly that window. The resulting window mask/pvals are embedded onto
    the full epoch time axis at ``[w0 : w0 + win_len]``. Then, only if a cluster
    reaches a window edge, the mask is grown out of the window through contiguous
    cluster-forming-threshold samples of a full-epoch permutation
    (``full_act`` vs ``full_bsl``) -- restoring the full temporal extent without
    changing which channels/clusters are significant.

    This is the routine time_perm_bands and the diff_* drivers used to duplicate
    inline; callers pass their own act/bsl arrays and ``tails`` and compute
    ``sig_ch_names`` from the returned window arrays.

    Parameters
    ----------
    win_act, win_bsl : np.ndarray
        Trial arrays for the windowed significance test (act vs bsl).
    full_act, full_bsl : np.ndarray
        Trial arrays for the full-epoch extent-growing test.
    n_channels, n_times : int
        Shape of the full epoch time axis to embed onto.
    w0 : int
        Index of the window's first sample on the full axis.
    p_thresh : float
        Cluster-forming threshold (also the pointwise suprathreshold for growth).
    tails : int
        1 (greater), -1 (less), or 2 (two-sided).

    Returns
    -------
    mask, pvals : np.ndarray, shape (n_channels, n_times)
        Full-axis significance mask and p-values.
    mask_w, pvals_w : np.ndarray
        The window-only mask/pvals (for the caller's sig-channel selection).
    """
    mask_w, pvals_w = time_perm_cluster(win_act, win_bsl,
                                        p_thresh=p_thresh,
                                        ignore_adjacency=ignore_adjacency,
                                        n_perm=n_perm, n_jobs=n_jobs,
                                        tails=tails)

    # Embed the window mask/pvals back onto the full epoch time axis.
    mask = np.zeros((n_channels, n_times), dtype=mask_w.dtype)
    pvals = np.ones((n_channels, n_times), dtype=pvals_w.dtype)
    w1 = w0 + mask_w.shape[1]  # one past the window's last sample
    mask[:, w0:w1] = mask_w
    pvals[:, w0:w1] = pvals_w

    # --- Extent: grow the saved mask out of the window in BOTH directions
    #     through contiguous cluster-forming-threshold samples over the full
    #     epoch. Skipped unless a cluster actually reaches a window edge. ---
    if mask[:, w0].any() or mask[:, w1 - 1].any():
        _, pvals_full = time_perm_cluster(full_act, full_bsl,
                                          p_thresh=p_thresh,
                                          ignore_adjacency=ignore_adjacency,
                                          n_perm=n_perm, n_jobs=n_jobs,
                                          tails=tails)
        supra = pvals_full < p_thresh  # pointwise suprathreshold, full-axis aligned
        for ch in np.flatnonzero(mask[:, w0:w1].any(axis=1)):
            # grow left from the window's leading edge
            if mask[ch, w0]:
                i = w0 - 1
                while i >= 0 and supra[ch, i]:
                    mask[ch, i] = True
                    pvals[ch, i] = pvals_full[ch, i]
                    i -= 1
            # grow right from the window's trailing edge
            if mask[ch, w1 - 1]:
                i = w1
                while i < supra.shape[1] and supra[ch, i]:
                    mask[ch, i] = True
                    pvals[ch, i] = pvals_full[ch, i]
                    i += 1

    return mask, pvals, mask_w, pvals_w
