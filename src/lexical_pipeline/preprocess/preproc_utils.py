"""Shared helpers for the preprocessing pipeline (denoise / apply_muscle).

Parameterized re-implementations of the functions the original single-file
``batch_preproc.py`` imported from ``utils.batch`` (``update_tsv`` /
``detect_outlier`` / ``load_eeg_chs`` / ``update_muscle_chs``).

Dataset-specific bits that used to live here as module constants (the per-subject
EEG/muscle data directories and the ``TRIGGER_NODROP`` table) now come from the
dataset config (``lexical_pipeline.config``); these helpers take the directories
as explicit arguments and the Trigger-drop policy is a config method.
"""
import os
import re
from os import path

import pandas as pd


def bids_task_tag(task_tag: str) -> str:
    """BIDS ``task-`` label = task tag with underscores removed.

    e.g. ``LexicalDecRepNoDelay`` -> ``LexicalDecRepNoDelay`` (no-op here, but
    matches ``utils.batch``'s ``task.replace('_', '')`` for tasks that contain
    underscores).
    """
    return task_tag.replace("_", "")


def channels_tsv_dir(bids_root: str, subject: str) -> str:
    """Directory that holds a subject's clean channels.tsv / events.tsv."""
    return path.join(bids_root, "derivatives", "clean", f"sub-{subject}", "ieeg")


def ensure_dir(*dirs: str) -> None:
    """``mkdir -p`` for one or more directories."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_eeg_chs(subject: str, eeg_dir: str) -> list:
    """Load the EEG/marker channel names to drop for a subject.

    Reads ``<eeg_dir>/<subject>_eeg_chans.csv`` (one channel name per line, no
    header; a single ``nan`` line means "no EEG channels").  Non-string entries
    (the ``nan`` placeholder) are kept here and filtered by the caller, matching
    the original behaviour.
    """
    csv_path = path.join(eeg_dir, f"{subject}_eeg_chans.csv")
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["eeg_chs"]
    return df["eeg_chs"].tolist()


def load_muscle_chs(subject: str, muscle_dir: str) -> list:
    """Load the muscle channel names for a subject.

    Reads ``<muscle_dir>/<subject>_muscle_chans.csv`` (one channel name per
    line, no header; a single ``nan`` line / empty file means "no muscle
    channels").  Returns the raw list (non-string ``nan`` filtered out).
    """
    csv_path = path.join(muscle_dir, f"{subject}_muscle_chans.csv")
    try:
        df = pd.read_csv(csv_path, header=None)
    except pd.errors.EmptyDataError:
        # An empty file is a valid "no muscle channels" marker (per spec).
        return []
    df.columns = ["muscle_chs"]
    return [c for c in df["muscle_chs"].tolist() if isinstance(c, str)]


def update_tsv(subj: str, search_dir: str, task_tag: str) -> None:
    """Drop boundary rows from each run's clean events.tsv (in place).

    Removes rows whose ``trial_type`` is "BAD boundary", "EDGE boundary" or
    "BAD_ACQ_SKIP" (artefacts of the line-noise filtering / cropping step).
    """
    task_tag_clean = bids_task_tag(task_tag)
    pattern = (
        f"sub-{subj}_task-{task_tag_clean}_acq-.+?_run-.+?_desc-clean_events.tsv"
    )
    files = [f for f in os.listdir(search_dir) if re.match(pattern, f)]
    if not files:
        raise ValueError(
            f"No clean events.tsv matching the pattern found for subj {subj} "
            f"in {search_dir}."
        )
    for file in files:
        input_file = path.join(search_dir, file)
        df = pd.read_csv(input_file, sep="\t")
        df_filtered = df[
            ~df["trial_type"].isin(
                ["BAD boundary", "EDGE boundary", "BAD_ACQ_SKIP"]
            )
        ]
        df_filtered.to_csv(input_file, sep="\t", index=False)
        print(f"Processed and replaced the original file: {input_file}")


def detect_outlier(subj: str, search_dir: str, task_tag: str) -> int:
    """Return 1 if any run's clean channels.tsv already marks an outlier.

    Used as a re-entrancy guard so denoise can be re-run idempotently without
    re-marking (which would compound bads).
    """
    task_tag_clean = bids_task_tag(task_tag)
    pattern = (
        f"sub-{subj}_task-{task_tag_clean}_acq-.+?_run-.+?_desc-clean_channels.tsv"
    )
    files = [f for f in os.listdir(search_dir) if re.match(pattern, f)]
    if not files:
        raise ValueError(
            f"No clean channels.tsv matching the pattern found for subj {subj} "
            f"in {search_dir}."
        )
    for file in files:
        data = pd.read_csv(path.join(search_dir, file), sep="\t")
        if (
            "status_description" in data.columns
            and "outlier" in data["status_description"].values
        ):
            return 1
    return 0


def update_muscle_chs(
    subj: str,
    search_dir: str,
    task_tag: str,
    muscle_dir: str,
) -> list:
    """Mark muscle channels as ``bad/muscle`` in each run's clean channels.tsv.

    Reads the subject's muscle CSV from ``muscle_dir`` and, for every channel
    listed, sets ``status=bad, status_description=muscle`` in every matching
    clean channels.tsv under ``search_dir``.  Idempotent (re-running does not
    compound).  Returns the list of muscle channels applied.
    """
    electrode_list = load_muscle_chs(subj, muscle_dir)

    task_tag_clean = bids_task_tag(task_tag)
    pattern = (
        f"sub-{subj}_task-{task_tag_clean}_acq-.+?_run-.+?_desc-clean_channels.tsv"
    )
    files = [f for f in os.listdir(search_dir) if re.match(pattern, f)]
    if not files:
        raise ValueError(
            f"No clean channels.tsv matching the pattern found for subj {subj} "
            f"in {search_dir}."
        )

    for file in files:
        file_path = path.join(search_dir, file)
        data = pd.read_csv(file_path, sep="\t")
        for electrode in electrode_list:
            if electrode in data["name"].values:
                idx = data[data["name"] == electrode].index[0]
                if (
                    data.at[idx, "status"] != "bad"
                    or data.at[idx, "status_description"] != "muscle"
                ):
                    data.at[idx, "status"] = "bad"
                    data.at[idx, "status_description"] = "muscle"
        data.to_csv(file_path, sep="\t", index=False)

    print(f"Updated files for subject {subj}: {files}")
    return electrode_list
