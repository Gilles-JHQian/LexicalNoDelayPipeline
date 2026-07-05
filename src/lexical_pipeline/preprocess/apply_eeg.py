"""Drop late-found EEG channels from the clean derivative (contract A).

The companion to ``denoise.py`` for EEG channels.  ``denoise`` drops the EEG /
marker channels known at denoise time (``<eeg_dir>/<subject>_eeg_chans.csv``)
*before* it writes ``derivatives/clean``.  Sometimes an EEG channel is only
spotted later (e.g. while marking muscle channels).  Rather than annotate it in
the channels.tsv, this step re-does the *same physical drop* denoise does, but
starting from the already-written clean derivative -- so the clean output stays
consistent with "EEG channels are simply not present".

Per subject:
1. Read ``derivatives/clean`` (all runs, concatenated) via ``raw_from_layout``.
2. From the subject's ``<eeg_dir>/<subject>_eeg_chans.csv`` list, find the
   channels still present in clean.  If none are present it is a no-op
   (idempotent -- already dropped by denoise or a previous run).
3. Snapshot the current channels.tsv ``status`` / ``status_description`` so the
   already-applied outlier / muscle marks survive the re-save (``save_derivative``
   rewrites channels.tsv from ``info['bads']`` and loses the descriptions).
4. Drop the EEG channels and re-save ``derivatives/clean`` (overwrite).
5. Strip the run-boundary rows the re-save re-introduces into events.tsv
   (same ``update_tsv`` denoise runs after saving), and restore the snapshotted
   channels.tsv marks onto the surviving channels.

``--subject all`` walks every ``*_eeg_chans.csv`` found in ``--eeg_dir``.
Dataset-specific values (bids_root/task/eeg_dir) come from ``dataset.toml``
(``--config``); any CLI flag overrides the config value.
"""
import argparse
import glob
import logging
import os
import re
import sys
from os import path

import pandas as pd
from bids.layout import BIDSLayout
from ieeg.io import raw_from_layout, save_derivative

from lexical_pipeline import config as _config
from .preproc_utils import (
    bids_task_tag,
    channels_tsv_dir,
    load_eeg_chs,
    update_tsv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _subjects_from_eeg_dir(eeg_dir: str) -> list:
    """All subject ids that have an EEG CSV in ``eeg_dir``."""
    files = sorted(glob.glob(path.join(eeg_dir, "*_eeg_chans.csv")))
    return [path.basename(f).replace("_eeg_chans.csv", "") for f in files]


def _clean_channels_files(search_dir: str, subject: str, task: str) -> list:
    """Full paths of a subject's per-run clean channels.tsv files."""
    task_tag_clean = bids_task_tag(task)
    pattern = (
        f"sub-{subject}_task-{task_tag_clean}_acq-.+?_run-.+?_desc-clean_channels.tsv"
    )
    return [
        path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if re.match(pattern, f)
    ]


def _snapshot_status(files: list) -> dict:
    """Capture ``{basename -> {ch_name -> (status, status_description)}}``.

    Taken before the re-save so the outlier / muscle marks (which live only in
    the channels.tsv, not in the raw object) can be restored afterwards.
    """
    snap = {}
    for fp in files:
        df = pd.read_csv(fp, sep="\t")
        has_desc = "status_description" in df.columns
        snap[path.basename(fp)] = {
            row["name"]: (
                row["status"],
                row["status_description"] if has_desc else "n/a",
            )
            for _, row in df.iterrows()
        }
    return snap


def _restore_status(files: list, snap: dict) -> None:
    """Re-apply snapshotted status / status_description onto surviving channels.

    ``save_derivative`` rewrites channels.tsv from ``info['bads']`` (so bad/good
    is preserved) but drops the specific ``status_description`` strings; dropped
    EEG channels are simply absent from the new file.  This puts the descriptions
    (outlier / muscle / ...) back for every channel that is still present.
    """
    for fp in files:
        namemap = snap.get(path.basename(fp))
        if not namemap:
            continue
        df = pd.read_csv(fp, sep="\t")
        if "status_description" not in df.columns:
            df["status_description"] = "n/a"
        for i, nm in enumerate(df["name"]):
            if nm in namemap:
                status, desc = namemap[nm]
                df.at[i, "status"] = status
                df.at[i, "status_description"] = desc
        df.to_csv(fp, sep="\t", index=False)


def apply_one(subject: str, task: str, bids_root: str, eeg_dir: str) -> None:
    search_dir = channels_tsv_dir(bids_root, subject)
    if not path.isdir(search_dir):
        logger.warning(
            "No clean derivative for %s at %s; skipping.", subject, search_dir
        )
        return

    try:
        eeg_list = [c for c in load_eeg_chs(subject, eeg_dir) if isinstance(c, str)]
    except FileNotFoundError:
        logger.warning(
            "No EEG channel CSV for %s in %s; skipping.", subject, eeg_dir
        )
        return

    # Load the clean derivative (all runs, concatenated) -- same read denoise's
    # outlier stage uses.
    layout = BIDSLayout(bids_root, derivatives=True)
    raw = raw_from_layout(
        layout.derivatives["derivatives/clean"],
        subject=subject,
        desc="clean",
        extension=".edf",
        preload=True,
    )

    present = [c for c in eeg_list if c in raw.ch_names]
    absent = [c for c in eeg_list if c not in raw.ch_names]
    if absent:
        logger.info(
            "EEG channels already absent from clean for %s (nothing to do for "
            "these): %s", subject, absent,
        )
    if not present:
        logger.info(
            "No listed EEG channels present in clean for %s; no re-save needed.",
            subject,
        )
        return

    # Snapshot channels.tsv marks BEFORE the re-save (outlier/muscle live only
    # in the tsv, not in the concatenated raw), then drop + re-save + restore.
    files_before = _clean_channels_files(search_dir, subject, task)
    snap = _snapshot_status(files_before)

    logger.info("Dropping %d EEG channel(s) from clean for %s: %s",
                len(present), subject, present)
    raw.drop_channels(present)

    out_layout = BIDSLayout(bids_root, derivatives=False)
    save_derivative(raw, out_layout, "clean", True)

    # The re-save regenerates events.tsv (re-introducing run-boundary rows) and
    # channels.tsv (losing status_description) -- mirror denoise's post-save
    # boundary strip, then restore the outlier/muscle marks.
    update_tsv(subject, search_dir, task)
    _restore_status(_clean_channels_files(search_dir, subject, task), snap)
    logger.info("Dropped EEG channels and re-saved clean for %s", subject)


def main(
    subject: str,
    task: str = None,
    bids_root: str = None,
    eeg_dir: str = None,
    config: str = None,
    **_,
) -> None:
    cfg = _config.load(config)
    task = task or cfg.input_task
    bids_root = bids_root or cfg.bids_root
    eeg_dir = eeg_dir or cfg.eeg_dir

    if subject.lower() == "all":
        subjects = _subjects_from_eeg_dir(eeg_dir)
        if not subjects:
            logger.warning("No *_eeg_chans.csv found in %s", eeg_dir)
        logger.info("Dropping EEG channels for %d subject(s) from %s",
                    len(subjects), eeg_dir)
    else:
        subjects = [subject]

    for subj in subjects:
        try:
            apply_one(subj, task, bids_root, eeg_dir)
        except (ValueError, FileNotFoundError) as e:
            logger.error("Failed for %s: %s", subj, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject", type=str, required=True,
        help="BIDS subject id, e.g. D0100, or 'all' to walk --eeg_dir",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task tag; default = dataset.toml [task].input_task",
    )
    parser.add_argument(
        "--bids_root", type=str, default=None,
        help="BIDS base whose derivatives/clean recording gets rewritten "
             "(default = dataset.toml [paths].bids_root)",
    )
    parser.add_argument(
        "--eeg_dir", type=str, default=None,
        help="Per-subject <S>_eeg_chans.csv dir "
             "(default = dataset.toml [paths].eeg_dir)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dataset.toml (default: $LEXPIPE_DATASET_CONFIG or upward search)",
    )
    args = parser.parse_args()
    main(**vars(args))
