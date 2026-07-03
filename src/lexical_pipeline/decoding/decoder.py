"""Decoding with optional pattern extraction via cross-validated permutation testing.

Extends the ``decode_permutation_scores`` interface from ``deocder.py`` with an
optional ``use_pattern`` flag.  When ``use_pattern=False`` the behaviour is
identical to the original function (returns accuracy-based obs_scores,
perm_scores, p_value).  When ``use_pattern=True`` the function additionally
extracts spatial patterns (via ``mne.decoding.get_coef``) on the observed
(non-permuted) CV folds only.  Pattern significance should be assessed
downstream via cross-fold t-test + cluster correction, not via permutation.

Requires the pipeline to wrap a linear model with ``LinearModel`` so that
``get_coef(..., 'patterns_', inverse_transform=True)`` works.
"""

import gc

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.base import clone
from joblib import Parallel, delayed
from mne.decoding import get_coef
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def feature_mixup(x_cls, alpha=1.0, rng=None):
    """Per-feature NaN interpolation using mixup.

    For each feature position (channel, time), NaN values across samples are
    replaced by a convex combination of two randomly chosen non-NaN values at
    that same feature position.  This is much more suitable than row-level
    mixup when every sample has *some* NaN but no single feature is entirely
    NaN across all samples.

    Parameters
    ----------
    x_cls : ndarray, shape (n_samples, ...)
        Data for one class.  First axis is the sample/epoch axis.
    alpha : float
        Beta distribution parameter (default 1.0 = uniform on [0, 1]).
    rng : int | np.random.RandomState | None
        Random state for reproducibility.

    Returns
    -------
    None – ``x_cls`` is modified **in-place**.
    """
    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(rng)

    n_samples = x_cls.shape[0]
    # flatten features: (n_samples, n_features)
    x_2d = x_cls.reshape(n_samples, -1)
    n_features = x_2d.shape[1]

    nan_mask = np.isnan(x_2d)
    if not nan_mask.any():
        return

    valid_mask = ~nan_mask
    valid_counts = valid_mask.sum(axis=0)  # (n_features,)

    # NaN cell coordinates (row = sample, col = feature)
    nan_rows, nan_cols = np.where(nan_mask)
    if nan_rows.size == 0:
        return
    col_counts = valid_counts[nan_cols]  # #valid samples in each NaN cell's column

    # --- Columns with 0 valid samples: fill NaNs with standard-normal noise ---
    zero_sel = col_counts == 0
    if zero_sel.any():
        n_zero = int(zero_sel.sum())
        x_2d[nan_rows[zero_sel], nan_cols[zero_sel]] = rng.normal(0, 1, n_zero)

    # --- Columns with >=1 valid sample: mixup gather ---
    # Build a flat lookup of valid (row) indices grouped by feature column so we
    # can map a uniform draw in [0, n_valid_col) to an actual valid row without a
    # per-feature Python loop (inverse-CDF gather via cumsum + offset).
    pos_sel = col_counts >= 1
    if pos_sel.any():
        # Build a flat table of valid sample-row indices grouped by feature.
        # np.where on the transpose iterates feature-major, so feat_idx is sorted
        # ascending and samp_idx[k] is a valid sample row for feature feat_idx[k].
        feat_idx, samp_idx = np.where(valid_mask.T)
        valid_rows_flat = samp_idx  # (total_valid,), blocked by feature
        # start offset of each feature's block within valid_rows_flat
        col_offsets = np.zeros(n_features + 1, dtype=np.int64)
        np.cumsum(valid_counts, out=col_offsets[1:])

        sel_rows = nan_rows[pos_sel]
        sel_cols = nan_cols[pos_sel]
        sel_counts = col_counts[pos_sel].astype(np.int64)
        sel_offsets = col_offsets[sel_cols]
        n_sel = sel_rows.size

        def _gather_valid():
            # draw a uniform valid-position per NaN cell, map to a real row
            u = (rng.random_sample(n_sel) * sel_counts).astype(np.int64)
            np.minimum(u, sel_counts - 1, out=u)  # guard against u == count
            return valid_rows_flat[sel_offsets + u]

        src1 = _gather_valid()
        src2 = _gather_valid()
        lam = rng.beta(alpha, alpha, size=n_sel)
        lam = np.maximum(lam, 1.0 - lam)  # ensure coefficient >= 0.5

        v1 = x_2d[src1, sel_cols]
        v2 = x_2d[src2, sel_cols]
        # Columns with exactly 1 valid sample: src1 == src2 == that sample, so the
        # convex combination collapses to a copy of the single valid value — no
        # special-casing needed.
        x_2d[sel_rows, sel_cols] = lam * v1 + (1.0 - lam) * v2


# reuse sample_fold from the existing decoder module
def sample_fold(
    X,
    y,
    train_idx,
    test_idx,
):
    """Sample a fold of data for cross-validation."""
    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = (y_train == cls)
        x_cls = X_train[idx]
        feature_mixup(x_cls, alpha=1.0, rng=42)
        X_train[idx] = x_cls

    # fill remaining test NaN with noise (seeded for run-to-run reproducibility)
    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        test_rng = np.random.RandomState(42)
        X_test[is_nan_test] = test_rng.normal(0, 1, int(np.sum(is_nan_test)))

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def decode_permutation_scores(
    X,
    y,
    cv,
    decoder,
    n_jobs: int = -1,
    n_permutations: int = 10,
    scoring: str = "accuracy",
    random_state: int = 42,
    use_pattern: bool = False,
):
    """Cross-validated permutation decoding with optional pattern extraction.

    When ``use_pattern=False`` (default), this function behaves identically to
    ``deocder.decode_permutation_scores``: it returns observed accuracy scores,
    permutation accuracy scores, and a p-value.

    When ``use_pattern=True``, observed spatial patterns are extracted via
    ``get_coef(pipeline, 'patterns_', inverse_transform=True)`` for each CV
    fold.  Permutation patterns are NOT computed (permutation creates an
    invalid null distribution for patterns).  Pattern significance should be
    assessed downstream via cross-fold t-test + cluster correction.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Neural time series data.
    y : ndarray, shape (n_epochs,)
        Class labels.
    cv : cross-validation splitter
        E.g. ``MinimumNaNSplit`` or ``StratifiedKFold``.
    decoder : sklearn estimator / pipeline
        The classification pipeline.  When ``use_pattern=True``, the pipeline
        must include ``LinearModel`` so that ``get_coef`` can extract patterns.
    n_jobs : int
        Number of parallel jobs (default ``-1`` = all cores).
    n_permutations : int
        Number of label permutations per fold (default 10).
    scoring : str
        Scoring metric name understood by ``sklearn.metrics.get_scorer``
        (default ``"accuracy"``).
    random_state : int
        Seed for reproducibility (default 42).
    use_pattern : bool
        If ``True``, also compute and return pattern-based results
        (default ``False``).

    Returns
    -------
    If ``use_pattern=False``:
        obs_scores : list[float]
            Observed score per CV fold.
        perm_scores : ndarray, shape (n_folds, n_permutations)
            Permutation scores per fold.
        p_value : float
            Proportion of mean-permutation scores >= mean observed score.

    If ``use_pattern=True``:
        obs_scores : list[float]
            Observed accuracy score per CV fold.
        perm_scores : ndarray, shape (n_folds, n_permutations)
            Permutation accuracy scores per fold.
        p_value : float
            Accuracy-based p-value.
        pattern_obs : ndarray, shape (n_folds, n_channels, n_times)
            Observed patterns per CV fold.
    """
    scorer = get_scorer(scoring)

    # --- NaN diagnostics --- (scan X once, reuse the boolean mask)
    nan_mask = np.isnan(X)
    feature_axes = tuple(range(1, X.ndim))
    nan_per_epoch_mask = nan_mask.any(axis=feature_axes)  # (n_epochs,)
    nan_per_epoch = nan_per_epoch_mask.sum()
    nan_per_channel = nan_mask.any(axis=(0, 2)).sum() if X.ndim == 3 else None
    total_nans = nan_mask.sum()
    logger.info("X shape: %s, y shape: %s", X.shape, y.shape)
    logger.info("Total NaN elements: %d / %d (%.2f%%)",
                total_nans, X.size, 100 * total_nans / X.size)
    logger.info("Epochs with any NaN: %d / %d", nan_per_epoch, X.shape[0])
    if nan_per_channel is not None:
        logger.info("Channels with any NaN: %d / %d", nan_per_channel, X.shape[1])
    for cls in np.unique(y):
        cls_mask = y == cls
        cls_nan = nan_per_epoch_mask[cls_mask].sum()
        logger.info("  Class %s: %d total, %d with NaN, %d clean",
                     cls, cls_mask.sum(), cls_nan, cls_mask.sum() - cls_nan)

    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")

    obs_scores = []
    perm_scores = []

    # pattern containers (only used when use_pattern=True)
    pattern_obs_list = [] if use_pattern else None

    for tr, te in tqdm(splits, desc="Cross-validation"):
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(X, y, tr, te)

        # ---- observed ----
        dec.fit(X_train, y_train)
        observed_score = scorer(dec, X_test, y_test)
        obs_scores.append(observed_score)

        if use_pattern:
            observed_pattern = get_coef(dec, "patterns_", inverse_transform=True)
            pattern_obs_list.append(observed_pattern)

        # ---- permutations ----
        rng_fold = np.random.RandomState(random_state)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        def one_perm(seed):
            r = np.random.RandomState(seed)
            y_train_perm = y_train.copy()
            r.shuffle(y_train_perm)
            dec_p = clone(dec)
            dec_p.fit(X_train, y_train_perm)
            acc = scorer(dec_p, X_test, y_test)
            return acc

        results_perm = Parallel(n_jobs=n_jobs)(
            delayed(one_perm)(s) for s in tqdm(seeds_fold, desc="Permutations")
        )

        perm_scores.append(np.asarray(results_perm))

        # Free per-fold intermediates
        del dec, X_train, X_test, y_train, y_test, results_perm
        gc.collect()

    # ---- aggregate accuracy ----
    score = np.mean(obs_scores)
    perm_scores = np.stack(perm_scores)  # (n_folds, n_permutations)

    # p-value (greater-is-better metric)
    p_value = (np.sum(perm_scores.mean(axis=0) >= score) + 1.0) / (n_permutations + 1.0)

    if not use_pattern:
        return obs_scores, perm_scores, p_value

    # ---- observed patterns only (no permutation patterns) ----
    pattern_obs = np.stack(pattern_obs_list)        # (n_folds, n_chn, n_times)

    return (
        obs_scores,
        perm_scores,
        p_value,
        pattern_obs,
    )


# ---------------------------------------------------------------------------
# Sign-flip testing
# ---------------------------------------------------------------------------

def decode_cv(X, y, cv, decoder, scoring="accuracy", use_pattern=False):
    """Cross-validate and return per-fold accuracy scores (and patterns).

    Unlike ``decode_permutation_scores``, this function does **not** run
    permutations.  Statistical significance is assessed downstream via
    ``sign_flip_test`` or ``sign_flip_cluster_1d``.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
    y : ndarray, shape (n_epochs,)
    cv : cross-validation splitter
    decoder : sklearn pipeline
        When *use_pattern* is ``True``, must contain ``LinearModel``.
    scoring : str
    use_pattern : bool

    Returns
    -------
    obs_scores : ndarray, shape (n_folds,)
    pattern_obs : ndarray, shape (n_folds, n_channels, n_times)
        Only returned when *use_pattern* is ``True``.
    """
    scorer = get_scorer(scoring)
    logger.info("decode_cv: X shape=%s, y shape=%s, use_pattern=%s",
                X.shape, y.shape, use_pattern)

    splits = list(cv.split(X, y))
    if not splits:
        raise ValueError("CV splitter produced no splits")

    obs_scores = []
    pattern_list = [] if use_pattern else None

    for tr, te in tqdm(splits, desc="Cross-validation"):
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(X, y, tr, te)
        dec.fit(X_train, y_train)
        obs_scores.append(scorer(dec, X_test, y_test))
        if use_pattern:
            pattern_list.append(
                get_coef(dec, "patterns_", inverse_transform=True)
            )
        del dec, X_train, X_test, y_train, y_test
        gc.collect()

    obs_scores = np.array(obs_scores)
    if use_pattern:
        return obs_scores, np.stack(pattern_list)
    return obs_scores


# ---------------------------------------------------------------------------
# Label-permutation testing for time-resolved accuracy
# ---------------------------------------------------------------------------

def cv_accuracy_resolved(X, y, cv, decoder, win_starts, window_samples,
                         scoring="accuracy"):
    """Mean cross-validated accuracy per sliding window for one label vector.

    This is the work-horse for the label-permutation null: pass a *shuffled*
    ``y`` and it returns the mean (across folds) decoding accuracy at every
    sliding window.  The same shuffled ``y`` is used for all windows, so the
    temporal autocorrelation structure of the null is preserved (required for a
    valid cluster-level test).

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
    y : ndarray, shape (n_epochs,)
        True or permuted labels.
    cv : cross-validation splitter
    decoder : sklearn pipeline
    win_starts : sequence[int]
        Start sample index of each sliding window.
    window_samples : int
        Window length in samples.
    scoring : str

    Returns
    -------
    acc : ndarray, shape (n_windows,)
        Mean fold accuracy at each window.
    """
    scorer = get_scorer(scoring)
    acc = np.empty(len(win_starts), dtype=float)
    for t_idx, start in enumerate(win_starts):
        Xw = X[..., start:start + window_samples]
        fold_scores = []
        for tr, te in cv.split(Xw, y):
            dec = clone(decoder)
            X_train, X_test, y_train, y_test = sample_fold(Xw, y, tr, te)
            dec.fit(X_train, y_train)
            fold_scores.append(scorer(dec, X_test, y_test))
            del dec, X_train, X_test, y_train, y_test
        acc[t_idx] = np.mean(fold_scores)
    return acc


def permutation_cluster_1d(obs, perm, p_thresh=0.05, cluster_alpha=0.05,
                           tails=1):
    """Cluster correction for 1-D (time) data from a label-permutation null.

    Each time point is standardised against the empirical permutation
    distribution (``z = (obs - null_mean) / null_std``).  Suprathreshold runs
    form clusters whose mass (sum of ``|z|``) is compared to the null
    distribution of maximum cluster mass built from the permutations
    themselves.  Standardising against the empirical null means the test is
    robust to the exact chance level (e.g. class imbalance), unlike a fixed
    ``1/n_classes`` reference.

    Parameters
    ----------
    obs : ndarray, shape (n_time,)
        Observed statistic per time point (e.g. mean accuracy across folds).
    perm : ndarray, shape (n_perm, n_time)
        Permutation null statistic per time point.
    p_thresh : float
        Cluster-forming threshold (converted to a z-critical value).
    cluster_alpha : float
        Cluster-level significance threshold.
    tails : {1, 2}
        1 = one-tailed (H1: obs > chance), 2 = two-tailed.

    Returns
    -------
    mask : ndarray, shape (n_time,), dtype bool
    p_corrected : ndarray, shape (n_time,)
    """
    from scipy.ndimage import label as nd_label
    from scipy.stats import norm

    obs = np.asarray(obs, dtype=float)
    perm = np.asarray(perm, dtype=float)
    n_perm, n_time = perm.shape

    null_mean = perm.mean(axis=0)
    null_std = np.maximum(perm.std(axis=0, ddof=1), 1e-12)
    z_obs = (obs - null_mean) / null_std
    z_perm = (perm - null_mean) / null_std

    z_crit = norm.ppf(1 - p_thresh) if tails == 1 else norm.ppf(1 - p_thresh / 2)

    def _clusters(z):
        sig = z > z_crit if tails == 1 else np.abs(z) > z_crit
        labeled, n_cl = nd_label(sig)
        masses, slices = [], []
        for ci in range(1, n_cl + 1):
            idx = labeled == ci
            masses.append(float(np.sum(np.abs(z[idx]))))
            pos = np.where(idx)[0]
            slices.append((pos[0], pos[-1] + 1))
        return slices, masses

    obs_slices, obs_masses = _clusters(z_obs)

    null_max = np.zeros(n_perm)
    for p in range(n_perm):
        _, masses = _clusters(z_perm[p])
        null_max[p] = max(masses) if masses else 0.0

    mask = np.zeros(n_time, dtype=bool)
    p_corrected = np.ones(n_time)
    for (start, end), mass in zip(obs_slices, obs_masses):
        p_cl = (1.0 + np.sum(null_max >= mass)) / (n_perm + 1.0)
        p_corrected[start:end] = p_cl
        if p_cl <= cluster_alpha:
            mask[start:end] = True

    return mask, p_corrected


def _generate_sign_flips(n_folds, max_exact=20):
    """Generate all sign-flip combinations.

    When *n_folds* <= *max_exact*, all ``2**n_folds`` combinations are
    enumerated exhaustively.  Otherwise, ``2**max_exact`` random
    combinations are drawn.

    Returns
    -------
    signs : ndarray, shape (n_flips, n_folds), values in {-1, +1}
    """
    if n_folds <= max_exact:
        n_flips = 2 ** n_folds
        # Vectorized: bit-shift to extract each bit position
        indices = np.arange(n_flips, dtype=np.int32)[:, np.newaxis]
        bits = np.arange(n_folds, dtype=np.int32)[np.newaxis, :]
        signs = np.where((indices >> bits) & 1, 1, -1)
    else:
        n_flips = 2 ** max_exact
        rng = np.random.RandomState(42)
        signs = rng.choice([-1, 1], size=(n_flips, n_folds))
    return signs


_SIGN_FLIP_BATCH = 64  # batch size to control memory in vectorized sign-flip


def sign_flip_test(fold_values, chance=0.0, tails=1):
    """Pointwise sign-flip test on fold-level values.

    Fully vectorized with batched broadcasting to control memory.

    Parameters
    ----------
    fold_values : ndarray, shape (n_folds, ...)
    chance : float
        Value subtracted before testing (e.g. ``1/n_classes`` for accuracy).
    tails : {1, 2}
        1 = one-tailed (H1: mean > chance).
        2 = two-tailed (H1: mean != chance).

    Returns
    -------
    p_value : float or ndarray, shape (...)
    """
    centered = np.asarray(fold_values, dtype=float) - chance
    n_folds = centered.shape[0]
    obs_mean = np.mean(centered, axis=0)

    signs_all = _generate_sign_flips(n_folds)  # (n_flips, n_folds)
    n_flips = signs_all.shape[0]
    extra_dims = centered.ndim - 1

    count = np.zeros_like(obs_mean, dtype=float)
    batch = _SIGN_FLIP_BATCH
    for start in range(0, n_flips, batch):
        end = min(start + batch, n_flips)
        # s_batch: (batch, n_folds, 1, 1, ...)
        s_batch = signs_all[start:end].reshape(
            end - start, n_folds, *([1] * extra_dims)
        )
        # null_means: (batch, ...)
        null_means = np.mean(centered[np.newaxis] * s_batch, axis=1)
        if tails == 2:
            count += np.sum(np.abs(null_means) >= np.abs(obs_mean), axis=0)
        else:
            count += np.sum(null_means >= obs_mean, axis=0)

    return count / n_flips


def sign_flip_cluster_1d(fold_values, chance, p_thresh=0.05,
                         cluster_alpha=0.05, tails=1):
    """Sign-flip cluster correction for 1-D (time) data.

    Uses the one-sample t-statistic at each time point for cluster forming,
    and cluster mass (sum of |t|) as the cluster-level statistic.  The null
    distribution of maximum cluster mass is built from exhaustive (or
    sampled) sign-flips.

    Parameters
    ----------
    fold_values : ndarray, shape (n_folds, n_time)
    chance : float
    p_thresh : float
        Cluster-forming threshold (converted to a t-critical value).
    cluster_alpha : float
        Cluster-level significance threshold.
    tails : {1, 2}

    Returns
    -------
    mask : ndarray, shape (n_time,), dtype bool
    p_corrected : ndarray, shape (n_time,)
    """
    from scipy.ndimage import label as nd_label
    from scipy.stats import t as t_dist

    centered = np.asarray(fold_values, dtype=float) - chance
    n_folds, n_time = centered.shape
    t_crit = t_dist.ppf(1 - p_thresh, df=n_folds - 1)
    signs_all = _generate_sign_flips(n_folds)  # (n_flips, n_folds)
    n_flips = signs_all.shape[0]
    sqrt_n = np.sqrt(n_folds)

    def _t_stat_batch(data_3d):
        """Vectorized t-stat: (batch, n_folds, n_time) -> (batch, n_time)."""
        m = np.mean(data_3d, axis=1)
        se = np.std(data_3d, axis=1, ddof=1) / sqrt_n
        se = np.maximum(se, 1e-12)
        return m / se

    def _clusters(t_vals):
        """Return list of (start, end) and list of cluster masses."""
        if tails == 1:
            sig = t_vals > t_crit
        else:
            sig = np.abs(t_vals) > t_crit
        labeled, n_cl = nd_label(sig)
        stats, slices = [], []
        for ci in range(1, n_cl + 1):
            idx = labeled == ci
            stats.append(np.sum(np.abs(t_vals[idx])))
            pos = np.where(idx)[0]
            slices.append((pos[0], pos[-1] + 1))
        return slices, stats

    # Observed t-stats
    obs_t = _t_stat_batch(centered[np.newaxis])[0]  # (n_time,)
    obs_slices, obs_stats = _clusters(obs_t)

    # Vectorized sign-flip: compute all t-stats in one batch
    # all_flipped: (n_flips, n_folds, n_time)
    all_flipped = centered[np.newaxis] * signs_all[:, :, np.newaxis]
    # all_t: (n_flips, n_time)
    all_t = _t_stat_batch(all_flipped)
    del all_flipped

    # Cluster extraction still needs a loop (nd_label is not vectorizable)
    null_max = np.zeros(n_flips)
    for i in range(n_flips):
        _, null_stats = _clusters(all_t[i])
        null_max[i] = max(null_stats) if null_stats else 0.0
    del all_t

    # Assign cluster-level p-values
    mask = np.zeros(n_time, dtype=bool)
    p_corrected = np.ones(n_time)
    for ci, (start, end) in enumerate(obs_slices):
        p_cl = np.mean(null_max >= obs_stats[ci])
        p_corrected[start:end] = p_cl
        if p_cl <= cluster_alpha:
            mask[start:end] = True

    return mask, p_corrected
