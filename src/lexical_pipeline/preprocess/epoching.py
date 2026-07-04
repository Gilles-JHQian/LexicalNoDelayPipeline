"""Generic epoching + high-gamma-power engine (dataset-agnostic).

Reusable helpers extracted from the LexicalDecRep NoDelay ``extract_ieeg_epochs``
script. The task-specific orchestration (which conditions/phases exist, the event
grammar, per-phase offsets, UP/Delay landmarks) lives in each dataset's driver,
which imports these helpers and passes its own ``output_task`` / ``bands`` /
``global_event_id`` / ``hierarchy_key``.
"""
import logging
import re

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_bids_path_from_fname
from ieeg.timefreq import gamma

logger = logging.getLogger(__name__)


def pick_event_name(epochs: mne.Epochs,
                    level: int = 0,
                    hierarchy_key: list = None) -> None:
    """
    Pick a specific level from the hierarchical event names and put the rest in metadata as a DataFrame.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to modify.
    level : int
        The index of the hierarchy level to use as the new event name (0-indexed).
    hierarchy_key : list, optional
        List of column names for the hierarchy levels.
        Default: ['event_type', 'task_type', 'stim_type', 'stim_content', 'resp_annotation']

    Notes
    -----
    For event names like "Cue/Repeat/Word/cat/CORRECT", this function:
    1. Extracts the specified level as the new simplified event name
    2. Stores remaining levels in epochs.metadata as a DataFrame
    """
    # Copy old event mapping to avoid in-place modification during iteration
    old_event_id = epochs.event_id.copy()
    old_events = epochs.events.copy()
    if hierarchy_key is None:
        hierarchy_key = ['event_type', 'task_type', 'stim_type', 'stim_content', 'resp_annotation']
    len_key = len(hierarchy_key)

    def collapse_name(name: str, len_key: int) -> tuple:
        parts = name.split("/")
        remain = parts.pop(level)
        tail = parts[len_key - 2:]
        tail = list(tail)
        parts = parts[:len_key - 2]
        parts.append(tail)
        return remain, parts

    # Create a copy to avoid modifying the original list during iteration
    hierarchy_key_copy = hierarchy_key.copy()
    hierarchy_key_copy.pop(level)

    groups = {}  # type: dict[str, set[int]]
    event_meta = {}  # type: dict[int, pd.DataFrame]
    for old_name, old_code in old_event_id.items():
        new_name, metadata = collapse_name(old_name, len_key)
        temp_df = pd.DataFrame(columns=hierarchy_key_copy)
        for i, temp_key in enumerate(hierarchy_key_copy):
            temp_df.loc[0, temp_key] = metadata[i]
        event_meta[old_code] = temp_df
        groups.setdefault(new_name, set()).add(int(old_code))

    # Merge each group: use mne.merge_events to combine codes into group's minimum value
    events = epochs.events  # (n_events, 3)
    if events is None or events.shape[1] < 3:
        raise ValueError("epochs.events is empty or has invalid shape, cannot merge.")

    # Construct metadata
    meta_rows = []
    for code in events[:, 2]:
        meta_rows.append(event_meta[code])

    meta_df = pd.concat(meta_rows, ignore_index=True)

    new_event_id = {}
    for new_name, old_codes in groups.items():
        code_list = sorted(old_codes)
        rep_code = code_list[0]  # use smallest id as representative

        if len(code_list) > 1:
            # Use mne.merge_events to merge all old codes in group into rep_code
            events = mne.merge_events(events=events,
                                      ids=code_list,
                                      new_id=rep_code,
                                      replace_events=True)
        # Record new unique mapping
        new_event_id[new_name] = rep_code

    # Write back updated events, event_id and metadata
    epochs.events = events
    epochs.event_id = new_event_id
    epochs.metadata = meta_df


def align_event_mapping(epochs: mne.Epochs,
                        global_event_id: dict) -> None:
    """
    Change the event_id to global event id dict and re-map the events matrix.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to be remapped.
    global_event_id : dict
        The global event id dict.
    Returns
    -------
    None
    """
    events = epochs.events
    local_event_id = epochs.event_id.copy()
    local_reverse = {v: k for k, v in local_event_id.items()}
    local_event_names = [local_reverse[code] for code in events[:, 2]]
    global_codes = [global_event_id[name] for name in local_event_names]
    events[:, 2] = global_codes
    epochs.events = events
    epochs.event_id = global_event_id


def interpolate(epochs, min_trials_per_class=2):
    """
    Fill NaNs per channel and class using class/global mean & std.

    - For each channel and class:
      * If class has >= min_trials_per_class valid samples -> fill with class mean (or mean+noise*std)
      * Else -> fill with global mean (or mean+noise*std)
    - If global has no donors at all (extremely rare), fill 0.0

    Parameters
    ----------
    epochs : mne.Epochs
    min_trials_per_class : int
        Threshold for using class-specific statistics
    """
    data = epochs._data  # shape (n_epochs, n_channels, n_times)
    if not np.any(np.isnan(data)):
        print("No NaN values found, skipping interpolation")
        return

    n_epochs, n_channels, n_times = data.shape
    cond_labels = epochs.events[:, 2]
    unique_classes = np.unique(cond_labels)

    total_nans = int(np.isnan(data).sum())
    print(f"Interpolating {total_nans} NaN values ({total_nans/(n_epochs*n_channels*n_times)*100:.2f}% of data)")

    for ch in range(n_channels):
        channel = data[:, ch, :]  # view (epochs, times)

        # get nonnan trials
        nan_trials = np.any(np.isnan(channel), axis=1)
        global_valid = channel[~nan_trials]
        g_mean = np.mean(global_valid, axis=0)
        g_std  = np.std(global_valid, axis=0)

        for c in unique_classes:
            rows = np.where(cond_labels == c)[0]
            sub = channel[rows, :]                 # (n_rows, times)
            sub_nan_trials = np.any(np.isnan(sub), axis=-1)

            # if no nan trials continue
            if np.sum(sub_nan_trials)==0:
                continue

            # Class donors for this channel (all non-NaN samples in this class)
            class_valid = sub[~sub_nan_trials]
            n_class_valid = class_valid.shape[0]

            if n_class_valid >= min_trials_per_class:
                c_mean = np.mean(class_valid, axis=0)
                c_std  = np.std(class_valid, axis=0)
                mean_to_use, std_to_use = c_mean, c_std
            else:
                mean_to_use, std_to_use = g_mean, g_std

            # Prepare replacement values for all NaNs in this class
            N = class_valid.shape[-1]

            for k, nan_trial in enumerate(sub_nan_trials):
                if nan_trial:
                    channel[rows[k], :] = mean_to_use + np.random.randn(N) * 1e-2 * std_to_use

        # Persist filled channel back
        data[:, ch, :] = channel
    epochs._data = data
    print(f"Interpolation complete. Remaining NaNs: {int(np.isnan(data).sum())}")

    return


def set_laplacian_reference(
    raw: mne.io.Raw
):
    """
    Apply a 1D Laplacian/bipolar re-reference along each electrode shaft.

    Overview
    - Channels are grouped by shaft using a name pattern like "LA1, LA2, ..." or
      "E1, E2, ...". The regex `([a-zA-Z]+)(\\d+)` extracts:
        - shaft name: alphabetic prefix (e.g., "LA", "E")
        - contact index: numeric suffix (e.g., 1, 2, ...)
    - For each shaft with >= 2 contacts:
        - First contact: V1 - V2
        - Last contact: VN - V(N-1)
        - Middle contacts: Vk - 0.5 * (V(k-1) + V(k+1)) (discrete Laplacian)
    """
    # Make a copy to avoid modifying original data
    raw = raw.copy()
    # Group channels by shafts
    shaft_groups = {}
    pattern = re.compile(r'([a-zA-Z]+)(\d+)')
    for ch_name in raw.ch_names:
        match = pattern.match(ch_name)
        if match:
            shaft_name = match.group(1) # The letter part is the shaft name
            contact_num = int(match.group(2))
            if shaft_name not in shaft_groups:
                shaft_groups[shaft_name] = []
            # Collect channel name under its shaft
            shaft_groups[shaft_name].append((ch_name, contact_num))
    for shaft_name, shaft_channels in shaft_groups.items():
        if len(shaft_channels) < 2:
            print(f"Skipping shaft '{shaft_name}' because it has fewer than 2 channels.")
            continue
        shaft_channels.sort(key=lambda x: x[1])
        shaft_channel_names = [ch[0] for ch in shaft_channels]

        # Store original data for this shaft to avoid in-place modification corruption
        shaft_indices = [raw.ch_names.index(ch_name) for ch_name in shaft_channel_names]
        original_data = raw._data[shaft_indices, :].copy()

        for i, current_ch_name in enumerate(shaft_channel_names):
            current_ch_idx = raw.ch_names.index(current_ch_name)
            if i == 0:
                # First contact: subtract next contact (bipolar)
                # V1 <- V1 - V2
                raw._data[current_ch_idx, :] = original_data[i, :] - original_data[i + 1, :]
            elif i == len(shaft_channel_names) - 1:
                # Last contact: subtract previous contact (bipolar)
                # VN <- VN - V(N-1)
                raw._data[current_ch_idx, :] = original_data[i, :] - original_data[i - 1, :]
            else:
                # Middle contacts: Laplacian (subtract average of neighbors)
                # Vk <- Vk - 0.5 * (V(k-1) + V(k+1))
                raw._data[current_ch_idx, :] = original_data[i, :] - 0.5 * (original_data[i - 1, :] + original_data[i + 1, :])
    return raw


def set_white_matter_reference(
    raw: mne.io.Raw
):
    """
    Set a white-matter (WM) reference using channels labeled as white matter.

    Procedure
    - Reads the BIDS `channels.tsv` sidecar corresponding to the current raw file.
    - Selects channels with `status_description == 'pure_white_matter'`.
      If none, falls back to `status_description == 'white_matter'`.
    - Applies MNE's referencing with the selected WM channels as reference.
    """
    # Locate the channels.tsv sidecar via BIDS path
    ref_path = get_bids_path_from_fname(raw.filenames[0])
    ref_path = ref_path.copy().update(suffix="channels", extension=".tsv").fpath

    # Read sidecar to discover WM channels
    ref_df = pd.read_csv(ref_path, sep='\t')
    # Prefer pure white matter channels; if none, use all white matter
    wm_channels = ref_df[ref_df['status_description'] == 'pure_white_matter']['name'].tolist()
    # Keep only those present in the current recording
    wm_channels = [ch for ch in wm_channels if ch in raw.ch_names]
    if not wm_channels:
        wm_channels = ref_df[ref_df['status_description'] == 'white_matter']['name'].tolist()

    # Infer channel type for referencing (e.g., 'ieeg' or 'eeg')
    ch_type = raw.get_channel_types(only_data_chs=True)[0]

    # Apply WM reference; MNE returns a new Raw instance
    raw = raw.set_eeg_reference(ref_channels=wm_channels, ch_type=ch_type)

    return raw


def save_epochs_and_bands(
    task_epochs, baseline_epoch, condition_, phase_,
    tmin, tmax, t_offset, fs,
    bids_layout, ref, subject, global_event_id,
    output_task, bands,
):
    """Save raw, bandpass-filtered, and power epochs for one phase-condition pair.

    ``output_task`` is the BIDS ``task-`` tag written into the output filenames
    (the per-dataset derivative tag). ``bands`` is a ``{name: (low, high)}`` map.
    """
    outpath = BIDSPath(
        root=bids_layout.root + f'/derivatives/epoch({ref})',
        subject=subject,
        task=output_task,
        datatype='epoch(raw)',
        check=False
    )
    outpath.mkdir(exist_ok=True)

    align_event_mapping(baseline_epoch, global_event_id)
    baseline_epoch.save(
        outpath.copy().update(
            suffix="raw", extension=".h5",
            description=condition_, processing="baseline", check=False
        ),
        overwrite=True
    )

    align_event_mapping(task_epochs, global_event_id)
    # Re-align the time axis so t=0 sits at the offset position (event marker +
    # t_offset) and the saved axis matches every other condition. Unlike the
    # cropped power epoch, the raw epoch keeps its +/-0.5 s padding -- it is the
    # wavelet edge buffer that time_perm_tfr crops off after the TFR -- so we
    # only relabel the axis (shift to start at tmin-0.5) rather than cropping.
    # The data samples are unchanged; for t_offset=0 this is a no-op.
    task_epochs.copy().shift_time(tmin - 0.5, relative=False).save(
        outpath.copy().update(
            suffix="raw", extension=".h5",
            processing=phase_, description=condition_, check=False
        ),
        overwrite=True
    )

    for band, freqs in bands.items():
        # Bandpass filter only (no Hilbert, no baseline)
        phase_task_band = task_epochs.copy()
        phase_task_band.filter(l_freq=freqs[0], h_freq=freqs[1], n_jobs=-1)
        # Same padding-preserving time-axis re-alignment as the raw epoch above.
        phase_task_band.shift_time(tmin - 0.5, relative=False)

        phase_out = outpath.copy().update(
            datatype='epoch(band)(raw)',
            suffix=f"{band}", extension=".h5", check=False
        )
        phase_out.mkdir(exist_ok=True)
        phase_task_band.save(
            phase_out.update(description=condition_, processing=phase_),
            overwrite=True
        )
        print(f"Saved unnormalized {band} epochs to {phase_out}")

        band_task = task_epochs.copy()
        band_baseline = baseline_epoch.copy()

        gamma.extract(band_task, passband=freqs, copy=False, n_jobs=-1)
        gamma.extract(band_baseline, passband=freqs, copy=False, n_jobs=-1)

        band_baseline = band_baseline.crop(tmin=-0.5, tmax=0)
        band_task = band_task.crop(tmin=tmin + t_offset, tmax=tmax + t_offset)
        # The cropped window was taken at [tmin + t_offset, tmax + t_offset]
        # (for passive Response t_offset = 0.8, giving an axis like (-0.2, 2.3)).
        # Re-label the time axis so its first sample is exactly tmin -- i.e.
        # t = 0 is pinned to the offset position (event marker + t_offset) and
        # the axis reads [tmin, tmax] = (-1, 1.5), matching every other
        # condition. This only relabels the time values; the data samples are
        # unchanged. For t_offset = 0 this is a no-op.
        band_task.shift_time(tmin, relative=False)

        print(f"Resampling {band} epochs to {fs} Hz")
        band_task = band_task.resample(sfreq=fs, n_jobs=-1)
        band_baseline = band_baseline.resample(sfreq=fs, n_jobs=-1)

        outpath.update(
            datatype='epoch(band)(power)',
            suffix=f"{band}", extension=".h5", check=False
        )
        outpath.mkdir(exist_ok=True)
        band_task.save(
            outpath.update(description=condition_, processing=phase_),
            overwrite=True
        )
        band_baseline.save(
            outpath.update(description=condition_, processing="baseline"),
            overwrite=True
        )
        print(f"Saved {band} epochs to {outpath}")
