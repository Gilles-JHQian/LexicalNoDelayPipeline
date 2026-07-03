"""Direct cross-domain decoding without CCA alignment.

This module implements cross-domain decoding for scenarios where the source and
target domains share the same electrode channels (e.g., perception vs production
from the same electrode group). No CCA alignment is needed.

Key components:
- ``DirectCrossDecoder``: Direct cross-domain decoder that trains on X1 and tests on X2
  without any feature space alignment.
- ``direct_cross_domain_permutation_scores``: Permutation testing for cross-domain decoding.

Typical workflow:
1. Load neural data from the same electrode group but different phases (e.g., perception/production).
2. Balance datasets to ensure equal trials per class.
3. Build an sklearn pipeline (e.g., Vectorizer -> StandardScaler -> PCA -> SVC).
4. Use ``DirectCrossDecoder`` and run permutation testing via ``direct_cross_domain_permutation_scores``.
"""


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import sys
import os
import h5py
from mne_bids import BIDSPath

from .decoder import feature_mixup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_roi_pair_with_intersection(
    bids_root,
    ref,
    train_roi,
    test_roi,
    train_phase,
    test_phase,
    band,
    datatype,
    train_tmin,
    train_tmax,
    test_tmin,
    test_tmax,
):
    """Load two ROIs and align them by intersecting channel names.

    This helper is intended for *direct* cross decoding where CCA is not used.
    It loads train/test ROIs (possibly different phases), crops to the specified
    time windows, reads ``chn_names`` from each file, takes the channel-name
    intersection, and returns data restricted to the common channels.

    Parameters
    ----------
    bids_root : str or Path
        Root directory of BIDS dataset.
    ref : str
        Reference type (e.g., 'bipolar', 'car'). Used to construct derivatives path.
    train_roi : str
        ROI name for training data (e.g., 'sensorimotor').
    test_roi : str
        ROI name for test data.
    train_phase : str
        Phase/description for training data (e.g., 'perception', 'production').
    test_phase : str
        Phase/description for test data.
    band : str
        Frequency band (e.g., 'highgamma').
    datatype : str
        Data type (e.g., 'phoneme').
    train_tmin, train_tmax : float
        Time window (in seconds) for training data.
    test_tmin, test_tmax : float
        Time window (in seconds) for test data.

    Returns
    -------
    X1 : ndarray, shape (n_epochs1, n_common_channels, n_times_train)
        Training data restricted to common channels.
    y1 : ndarray, shape (n_epochs1,)
        Training labels.
    X2 : ndarray, shape (n_epochs2, n_common_channels, n_times_test)
        Test data restricted to common channels.
    y2 : ndarray, shape (n_epochs2,)
        Test labels.
    common_ch_names : ndarray, shape (n_common_channels,)
        Array of common channel names (strings), sorted as in train ROI.
    """

    # Construct BIDS paths for derivatives decoding data (same convention as cross_decoder)
    root = BIDSPath(
        root=os.path.join(bids_root, "derivatives", f"decoding({ref})"),
        datatype=datatype,
        suffix=band,
        extension=".h5",
        check=False,
    )

    # Train ROI file
    train_path = root.copy().update(subject=train_roi, description=train_phase)
    train_files = train_path.match()
    if not train_files:
        raise FileNotFoundError(f"No files found for train ROI {train_roi} ({train_phase})")

    # Test ROI file
    test_path = root.copy().update(subject=test_roi, description=test_phase)
    test_files = test_path.match()
    if not test_files:
        raise FileNotFoundError(f"No files found for test ROI {test_roi} ({test_phase})")

    # Load train data
    # right now just take the first phoneme
    with h5py.File(train_files[0], "r") as f_tr:
        X1 = f_tr["X"][()]
        y1 = f_tr["y"][()]
        fs_tr = f_tr.attrs["fs"]
        tmin_tr = f_tr.attrs["tmin"]
        tmax_tr = f_tr.attrs["tmax"]
        chn_tr = f_tr["chn_names"][()]

    # Load test data
    with h5py.File(test_files[0], "r") as f_te:
        X2 = f_te["X"][()]
        y2 = f_te["y"][()]
        fs_te = f_te.attrs["fs"]
        tmin_te = f_te.attrs["tmin"]
        tmax_te = f_te.attrs["tmax"]
        chn_te = f_te["chn_names"][()]

    # Sanity check on sampling frequency
    if fs_tr != fs_te:
        raise ValueError(f"Sampling rate mismatch: train fs={fs_tr}, test fs={fs_te}")

    fs = fs_tr

    # Crop in time for train data
    # take the 0.2s after 0s
    start_tr = int(fs * (train_tmin - tmin_tr))
    end_tr = int(fs * (train_tmax - tmin_tr))
    X1 = X1[:, :, start_tr:end_tr]

    # Crop in time for test data
    start_te = int(fs * (test_tmin - tmin_te))
    end_te = int(fs * (test_tmax - tmin_te))
    X2 = X2[:, :, start_te:end_te]

    # Align channels by intersection of channel names
    chn_tr = np.asarray(chn_tr).astype(str)
    chn_te = np.asarray(chn_te).astype(str)

    common_ch, idx_tr, idx_te = np.intersect1d(chn_tr, chn_te, return_indices=True)
    if common_ch.size == 0:
        raise ValueError(
            f"No overlapping channels between train ROI {train_roi} and test ROI {test_roi}."
        )

    # Restrict to common channels, following the order of train ROI
    X1 = X1[:, idx_tr, :]
    X2 = X2[:, idx_te, :]

    logger.info(
        "Loaded ROIs with channel intersection: train=%s (%d ch), test=%s (%d ch), common=%d ch",
        train_roi,
        len(chn_tr),
        test_roi,
        len(chn_te),
        len(common_ch),
    )

    return X1, y1, X2, y2, common_ch


def sample_fold(X, y, train_idx, test_idx):
    """Sample a fold of data for cross-validation.
    
    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Neural data.
    y : ndarray, shape (n_epochs,)
        Labels.
    train_idx : array-like
        Indices for training set.
    test_idx : array-like
        Indices for test set.
        
    Returns
    -------
    X_train, X_test, y_train, y_test : ndarrays
        Split data and labels.
    """
    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx].copy(), y[test_idx].copy()
    
    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = (y_train == cls)
        x_cls = X_train[idx]
        feature_mixup(x_cls, alpha=1.0, rng=42)
        X_train[idx] = x_cls
    
    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        X_test[is_nan_test] = np.random.normal(0, 1, int(np.sum(is_nan_test)))
    
    return X_train, X_test, y_train, y_test


class DirectCrossDecoder(BaseEstimator, ClassifierMixin):
    """Direct cross-domain neural decoder without feature alignment.

    This class implements cross-domain decoding for scenarios where the source
    and target domains share the same electrode channels. It trains a classifier
    on the source domain (X1) and tests on the target domain (X2) directly.
    
    Use this when:
    - X1 and X2 come from the same electrode group (same channels)
    - Only the experimental phase differs (e.g., perception vs production)

    Parameters
    ----------
    estimator : Pipeline
        Scikit-learn compatible estimator pipeline (e.g., Vectorizer + PCA + SVM).
        Must implement `fit` and `predict` methods.
    random_state : int, default=42
        Random seed for reproducible results.
        
    Attributes
    ----------
    estimator : Pipeline
        Fitted classifier pipeline.
        
    Examples
    --------
    >>> from mne.decoding import Vectorizer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC, LinearSVC
    >>> 
    >>> estimator = make_pipeline(
    ...     Vectorizer(),
    ...     StandardScaler(),
    ...     PCA(n_components=0.85),
    ...     SVC(kernel='linear')
    ... )
    >>> 
    >>> decoder = DirectCrossDecoder(estimator)
    >>> decoder.fit(X1, y1, X2, y2)  # X2, y2 not used but kept for API compatibility
    >>> predictions = decoder.predict(X2_test)
    """
     
    def __init__(
        self,
        estimator: Pipeline,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.estimator = estimator
         
    def fit(self, X1, y1, X2=None, y2=None):
        """Fit the decoder on source domain data.
        
        Parameters
        ----------
        X1 : ndarray, shape (n_epochs, n_channels, n_times)
            Source domain neural data for training.
        y1 : ndarray, shape (n_epochs,)
            Class labels for X1.
        X2 : ndarray, optional
            Target domain data (not used, kept for API compatibility with CrossDecoder).
        y2 : ndarray, optional
            Target domain labels (not used, kept for API compatibility).

        Returns
        -------
        self : DirectCrossDecoder
            Fitted decoder.
        """
        if X1.ndim != 3:
            raise ValueError("X1 must be 3D array in (epoch, channel, time) format")
        
        self.estimator.fit(X1, y1)
        return self
        
    def predict(self, X2):
        """Predict class labels for target domain data.
        
        Parameters
        ----------
        X2 : ndarray, shape (n_epochs, n_channels, n_times)
            Target domain neural data to classify.

        Returns
        -------
        predicted : ndarray, shape (n_epochs,)
            Predicted class labels.
        """
        return self.estimator.predict(X2)
    
    def score(self, X, y):
        """Return accuracy score on given data.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Neural data to score.
        y : ndarray, shape (n_epochs,)
            True labels.
            
        Returns
        -------
        score : float
            Accuracy score.
        """
        return self.estimator.score(X, y)


def direct_cross_domain_permutation_scores(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    scoring: str = "accuracy",
    n_permutations: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """Cross-domain permutation test without CCA alignment.
    
    For each CV fold:
    1. Train classifier on X1_train
    2. Test on X2_test
    3. Build permutation baseline by shuffling y1_train
    
    Parameters
    ----------
    X1 : ndarray, shape (n_epochs, n_channels, n_times)
        Source domain neural data (train).
    y1 : ndarray, shape (n_epochs,)
        Source domain labels.
    X2 : ndarray, shape (n_epochs, n_channels, n_times)
        Target domain neural data (test).
    y2 : ndarray, shape (n_epochs,)
        Target domain labels.
    cv : CV splitter
        Sklearn-compatible cross-validation splitter.
    cross_decoder : DirectCrossDecoder
        Configured DirectCrossDecoder instance.
    scoring : str, default='accuracy'
        Scoring metric name.
    n_permutations : int, default=10
        Number of label permutations per fold.
    n_jobs : int, default=-1
        Parallel jobs for permutations.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    obs_scores : list
        Observed scores per fold.
    perm_scores : ndarray, shape (n_folds, n_permutations)
        Permutation scores per fold.
    p_value : float
        One-sided p-value (proportion of permutations >= observed mean).
    """
    scorer = get_scorer(scoring)
    splits = list(cv.split(X1, y1))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    obs_scores = []
    perm_scores = []
    
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        dec = clone(cross_decoder)
        
        X1_train, X1_test, y1_train, y1_test = sample_fold(X1, y1, tr, te)
        X2_train, X2_test, y2_train, y2_test = sample_fold(X2, y2, tr, te)
        
        # Fit on X1_train, score on X2_test
        dec.fit(X1_train, y1_train)
        observed_score = scorer(dec, X2_test, y2_test)
        obs_scores.append(observed_score)
        
        # Permutation test - generate different seeds per fold
        rng_fold = np.random.RandomState(random_state + fold_idx)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        def one_perm(seed):
            r = np.random.RandomState(seed)
            y1_train_perm = y1_train.copy()
            r.shuffle(y1_train_perm)
            est_p = clone(dec.estimator)
            est_p.fit(X1_train, y1_train_perm)
            return scorer(est_p, X2_test, y2_test)

        perm_score = np.asarray(
            Parallel(n_jobs=n_jobs)(
                delayed(one_perm)(s) for s in tqdm(seeds_fold, desc="Permutations", leave=False)
            )
        )
        perm_scores.append(perm_score)
        
    score = np.mean(obs_scores)
    perm_scores = np.stack(perm_scores)
    
    # p-value (greater is better metric)
    p_value = (np.sum(perm_scores.mean(axis=0) >= score) + 1.0) / (n_permutations + 1.0)
    
    return obs_scores, perm_scores, p_value


def direct_cross_domain_resolved_permutation_scores(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    scoring: str = "accuracy",
    n_permutations: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    window: float = 0.2,
    step: float = 0.1,
    fs: int = 128,
    tmin: float = 0,
    tmax: float = 0.5,
):
    """Time-resolved cross-domain decoding without CCA, with permutation testing and FDR.

    For each CV fold, this function:
    - Slides a window over the temporal axis [tmin, tmax] with length ``window`` and step ``step``.
    - For each time window, trains the estimator on X1 (train) and evaluates on X2 (test) directly.
    - Builds a permutation baseline by shuffling y1 in the training split ``n_permutations`` times.

    After all folds finish, observed fold scores and permuted fold scores are averaged across folds
    to obtain per-time observed statistics and the corresponding permutation distribution. Per-time
    one-sided p-values are computed and corrected using FDR-BH.

    Parameters
    ----------
    X1, X2 : ndarray, shape (n_epochs, n_channels, n_times)
        Source and target domain data (same electrode group, different phases).
    y1, y2 : ndarray, shape (n_epochs,)
        Class labels. After balancing upstream, they should be aligned across domains.
    cv : CV splitter
        Any sklearn-compatible splitter with ``split(X1, y1)`` yielding (train_idx, test_idx).
    cross_decoder : DirectCrossDecoder
        A configured ``DirectCrossDecoder`` instance wrapping an sklearn estimator pipeline.
    scoring : str, default='accuracy'
        Scorer name accepted by sklearn's ``get_scorer``.
    n_permutations : int, default=10
        Number of label permutations per fold per time window.
    n_jobs : int, default=-1
        Parallel jobs for permutation evaluations within each time window.
    random_state : int, default=42
        Base seed for reproducibility. Seeds are expanded per fold/time internally.
    window : float, default=0.2
        Window length (seconds) for time-resolved decoding.
    step : float, default=0.1
        Step size (seconds) for sliding window.
    fs : int, default=128
        Sampling rate used to convert seconds to samples.
    tmin, tmax : float, default=(0, 0.5)
        Temporal range (seconds) over which windows are evaluated.

    Returns
    -------
    obs_scores : ndarray, shape (T, n_folds)
        Observed accuracy (or other ``scoring``) per time window per fold.
    perm_scores : ndarray, shape (T, n_permutations, n_folds)
        Permutation baseline scores per time window, permutation, and fold.
    pvals_fdr : ndarray, shape (T,)
        FDR-BH corrected per-time p-values (one-sided; greater-is-better).
    """
    from statsmodels.stats.multitest import multipletests
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X1, y1))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    n_wins = int(np.floor((tmax - tmin - window) / step)) + 1
    time_points = tmin + window + np.arange(n_wins) * step
    window_samples = int(window * fs)
    
    n_folds = len(splits)
    obs_scores = np.empty((len(time_points), n_folds))
    perm_scores = np.empty((len(time_points), n_permutations, n_folds))
    
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        
        dec = clone(cross_decoder)
        
        X1_train, X1_test, y1_train, y1_test = sample_fold(X1, y1, tr, te)
        X2_train, X2_test, y2_train, y2_test = sample_fold(X2, y2, tr, te)

        # Generate permutation seeds once per fold
        rng_fold = np.random.RandomState(random_state + fold_idx)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        # time resolved decoding (no CCA - use X1/X2 directly)
        for t_idx, time_end in enumerate(time_points):
            
            end_sample = int((time_end - tmin) * fs) + 1
            start_sample = end_sample - window_samples
            
            if start_sample < 0 or end_sample > X1_train.shape[-1]:
                logger.warning(f"Window out of bounds for time {time_end:.3f}s, skipping")
                continue
            
            x1_train_s = X1_train[..., start_sample:end_sample]
            x2_test_s = X2_test[..., start_sample:end_sample]
            
            dec.estimator.fit(x1_train_s, y1_train)
            observed_score = scorer(dec.estimator, x2_test_s, y2_test)

            def one_perm(seed):
                r = np.random.RandomState(seed)
                y1_train_perm = y1_train.copy()
                r.shuffle(y1_train_perm)
                est_p = clone(dec.estimator)
                est_p.fit(x1_train_s, y1_train_perm)
                return scorer(est_p, x2_test_s, y2_test)

            perm_score = np.asarray(Parallel(n_jobs=n_jobs, batch_size=40)(delayed(one_perm)(s) for s in seeds_fold))
            
            obs_scores[t_idx, fold_idx] = observed_score
            perm_scores[t_idx, :, fold_idx] = perm_score
            
    observed_t = obs_scores.mean(axis=1)     # (T,)
    perm_t = perm_scores.mean(axis=2)        # (T, n_perm)

    pvals_pt = ((perm_t >= observed_t[:, None]).sum(axis=1) + 1) / (perm_t.shape[1] + 1)
    _, pvals_fdr, _, _ = multipletests(pvals_pt, alpha=0.05, method='fdr_bh')

    return obs_scores, perm_scores, pvals_fdr


def direct_cross_domain_generalized_permutation_scores(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    scoring: str = "accuracy",
    n_permutations: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    window: float = 0.2,
    step: float = 0.1,
    fs: int = 128,
    train_tmin: float = 0,
    train_tmax: float = 0.5,
    test_tmin: float = 0,
    test_tmax: float = 0.5,
):
    """Temporal generalization (train-time × test-time) decoding without CCA, with permutations and FDR.

    For each CV fold, this function:
    - Constructs two sliding-window grids: a train-time grid within [train_tmin, train_tmax] and
      a test-time grid within [test_tmin, test_tmax], each with window length ``window`` and step ``step``.
    - For every cell (t_train, t_test), trains the estimator on X1 (training split) using the
      train window and evaluates on X2 (test split) using the test window directly (no CCA).
    - Builds a permutation baseline at each cell by shuffling y1 in the training split ``n_permutations`` times.

    After all folds finish, scores are averaged across folds to obtain a 2D observed map and a corresponding
    permutation distribution per cell. One-sided per-cell p-values are computed from the permutation null and
    then corrected across the 2D field using FDR-BH.

    Parameters
    ----------
    X1, X2 : ndarray, shape (n_epochs, n_channels, n_times)
        Source (train) and target (test) domain data (same electrode group, different phases).
    y1, y2 : ndarray, shape (n_epochs,)
        Class labels for X1/X2. Upstream balancing should align label distributions.
    cv : CV splitter
        Any sklearn-compatible splitter yielding (train_idx, test_idx) on (X1, y1).
    cross_decoder : DirectCrossDecoder
        Configured direct cross-decoder (estimator pipeline, no CCA).
    scoring : str, default='accuracy'
        Scorer name accepted by sklearn's ``get_scorer``.
    n_permutations : int, default=10
        Number of label permutations per fold per (t_train, t_test) cell.
    n_jobs : int, default=-1
        Parallel jobs for permutation evaluations within each cell.
    random_state : int, default=42
        Base seed for reproducibility (expanded per fold/cell internally as needed).
    window : float, default=0.2
        Window length in seconds for temporal slicing.
    step : float, default=0.1
        Step size in seconds between adjacent windows.
    fs : int, default=128
        Sampling rate to convert seconds to sample indices.
    train_tmin, train_tmax : float, default=(0, 0.5)
        Temporal range (seconds) for train-time windows (applied to X1 on the training split).
    test_tmin, test_tmax : float, default=(0, 0.5)
        Temporal range (seconds) for test-time windows (applied to X2 on the test split).

    Returns
    -------
    obs_scores : ndarray, shape (T_train, T_test, n_folds)
        Observed score (``scoring``) per (train-time, test-time) cell and per fold.
    perm_scores : ndarray, shape (T_train, T_test, n_permutations, n_folds)
        Permutation baseline scores per cell, permutation, and fold.
    pvals_fdr : ndarray, shape (T_train, T_test)
        FDR-BH corrected per-cell p-values (one-sided; greater-is-better).
    """
    scorer = get_scorer(scoring)
    splits = list(cv.split(X1, y1))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    # format to .2f 
    train_tmin, train_tmax, test_tmin, test_tmax = [round(t, 2) for t in [train_tmin, train_tmax, test_tmin, test_tmax]]
    
    n_train = int(np.floor((train_tmax - train_tmin - window) / step)) + 1
    train_time_points = train_tmin + window + np.arange(n_train) * step
    n_test = int(np.floor((test_tmax - test_tmin - window) / step)) + 1
    test_time_points = test_tmin + window + np.arange(n_test) * step
    window_samples = int(window * fs)
    
    n_folds = len(splits)
    obs_scores = np.empty((len(train_time_points), len(test_time_points), n_folds))
    perm_scores = np.empty((len(train_time_points), len(test_time_points), n_permutations, n_folds))
    
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        
        dec = clone(cross_decoder)
        
        X1_train, X1_test, y1_train, y1_test = sample_fold(X1, y1, tr, te)
        X2_train, X2_test, y2_train, y2_test = sample_fold(X2, y2, tr, te)

        # Generate permutation seeds once per fold (shared across all time windows)
        rng_fold = np.random.RandomState(random_state + fold_idx)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        # time generalized decoding - batch permutation fits per train_time
        for train_t_idx, train_time_end in enumerate(train_time_points):
            
            end_train = int(round((train_time_end - train_tmin) * fs))
            start_train = end_train - window_samples
            
            if start_train < 0 or end_train > X1_train.shape[-1]:
                logger.warning(f"Window out of bounds for time {train_time_end:.3f}s, skipping")
                continue
            
            x1_train_s = X1_train[..., start_train:end_train]
            
            # Fit observed estimator once for this train_time
            dec.estimator.fit(x1_train_s, y1_train)
            
            # Pre-fit all permutation estimators for this train_time (batch optimization)
            def fit_one_perm(seed):
                r = np.random.RandomState(seed)
                y1_train_perm = y1_train.copy()
                r.shuffle(y1_train_perm)
                est_p = clone(dec.estimator)
                est_p.fit(x1_train_s, y1_train_perm)
                return est_p
            
            perm_estimators = Parallel(n_jobs=n_jobs, batch_size=40)(
                delayed(fit_one_perm)(s) for s in seeds_fold
            )
            
            # Now evaluate on all test_times (only predict, no fit - much faster)
            for test_t_idx, test_time_end in enumerate(test_time_points):
                
                end_test = int(round((test_time_end - test_tmin) * fs))
                start_test = end_test - window_samples

                if start_test < 0 or end_test > X2_test.shape[-1]:
                    obs_scores[train_t_idx, test_t_idx, fold_idx] = np.nan
                    perm_scores[train_t_idx, test_t_idx, :, fold_idx] = np.nan
                    continue
            
                x2_test_s = X2_test[..., start_test:end_test]
                
                # Observed score
                observed_score = scorer(dec.estimator, x2_test_s, y2_test)
                obs_scores[train_t_idx, test_t_idx, fold_idx] = observed_score
                
                # Permutation scores (only scoring, estimators already fitted)
                perm_score = np.array([scorer(est_p, x2_test_s, y2_test) for est_p in perm_estimators])
                perm_scores[train_t_idx, test_t_idx, :, fold_idx] = perm_score
            
    observed_mean = obs_scores.mean(axis=-1)        # (Ttr, Tte)
    perm_mean = perm_scores.mean(axis=-1)           # (Ttr, Tte, n_perm)

    P = n_permutations

    # --- 2D cluster-based permutation correction ---
    from scipy.ndimage import label as nd_label

    # Raw per-cell p-values
    pvals_raw = ((perm_mean >= observed_mean[..., None]).sum(axis=2) + 1) / (P + 1)

    cluster_p_thresh = 0.05
    cluster_alpha = 0.05

    # Out-of-bounds (train, test) cells were filled with NaN. NaN comparisons
    # return False, so those cells get a spuriously low p-value (~1/(P+1)) and
    # would otherwise be marked significant and merged into clusters. Exclude
    # them explicitly from both the observed and the null cluster maps. (No-op
    # for the current symmetric train/test grids, but required as soon as a
    # non-symmetric [tmin, tmax] is used.)
    valid = ~np.isnan(observed_mean)
    pvals_raw[~valid] = 1.0

    # Observed binary map and clusters
    obs_sig = (pvals_raw <= cluster_p_thresh) & valid
    obs_labeled, obs_n_cl = nd_label(obs_sig)
    obs_cluster_masses = []
    for ci in range(1, obs_n_cl + 1):
        cl_mask = obs_labeled == ci
        obs_cluster_masses.append(cl_mask.sum())

    # Build null distribution of max cluster size from permutations.
    # For each permutation p, treat that permutation as the "observed"
    # and compute per-cell p-values against the remaining permutations.
    null_max_cluster = np.zeros(P)
    for pi in range(P):
        # This permutation's score map
        perm_obs = perm_mean[:, :, pi]                        # (Ttr, Tte)
        # p-value of this permutation against the *remaining* P-1 permutations.
        # Counting all P permutations >= perm_obs includes the self-comparison
        # (perm_obs >= perm_obs is always True), so it equals the rest-count + 1.
        # Hence ((rest_count) + 1) / P == count_all / P — exactly, no np.delete.
        count_all = (perm_mean >= perm_obs[..., None]).sum(axis=2)  # includes self
        perm_p = count_all / P
        perm_sig = (perm_p <= cluster_p_thresh) & valid
        perm_labeled_i, perm_n_cl = nd_label(perm_sig)
        if perm_n_cl == 0:
            null_max_cluster[pi] = 0
        else:
            # cluster sizes via bincount over labels (0 = background)
            sizes = np.bincount(perm_labeled_i.ravel())[1:]
            null_max_cluster[pi] = sizes.max()

    # Assign cluster-level p-values
    pvals_corrected = np.ones_like(pvals_raw)
    mask = np.zeros_like(pvals_raw, dtype=bool)
    for ci in range(1, obs_n_cl + 1):
        cl_mask = obs_labeled == ci
        cl_mass = obs_cluster_masses[ci - 1]
        cl_p = (np.sum(null_max_cluster >= cl_mass) + 1) / (P + 1)
        pvals_corrected[cl_mask] = cl_p
        if cl_p <= cluster_alpha:
            mask[cl_mask] = True

    logger.info(f"  2D cluster correction: {obs_n_cl} observed clusters, "
                f"{mask.sum()} significant cells")

    return obs_scores, perm_scores, pvals_corrected