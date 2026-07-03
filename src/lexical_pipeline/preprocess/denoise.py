"""Denoise iEEG and produce the ``derivatives/clean`` BIDS derivative.

This is the "generate clean" half of the lab preprocessing pipeline, extracted
from the monolithic ``batch_preproc.py`` (the ``linernoise`` stage L129-198 +
the ``outlierchs`` stage L200-232) into a single, per-subject, sbatch-friendly
script.  The heavy wavelet / spectrum / muscle-marking work lives in the sister
Muscle tool and is intentionally NOT here.

Pipeline (equivalent to the original, per subject):
1. Read raw iEEG from a BIDS layout.
2. Drop EEG + Trigger/marker channels (per-subject EEG list + Trigger rule).
3. Line-noise filter twice (notch 60 / 120 / 180 / 240 Hz).
4. (crop_empty subjects only) crop empty data.
5. Save as ``derivatives/clean`` (EDF, no CAR; bad channels keep their signal,
   only the channels.tsv is annotated) and strip boundary rows from events.tsv.
6. Re-read clean, mark outlier channels and write ``bad/outlier`` into
   channels.tsv.  Idempotent: skipped if already done.

Dataset-specific values (bids_root/raw_root/task/eeg_dir, the Trigger-drop and
crop-empty subject sets, and the line-filter / outlier numeric params) come from
``dataset.toml`` (``--config``); any CLI flag overrides the config value.
"""
import argparse
import logging
import sys
from os import path

from bids.layout import BIDSLayout
from ieeg.io import raw_from_layout, save_derivative, update
from ieeg.mt_filter import line_filter
from ieeg.navigate import crop_empty_data, channel_outlier_marker

from lexical_pipeline import config as _config
from .preproc_utils import (
    channels_tsv_dir,
    detect_outlier,
    ensure_dir,
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


def linenoise(
    subject: str,
    task: str,
    raw_root: str,
    bids_root: str,
    eeg_dir: str,
    cfg,
) -> None:
    """Drop EEG/Trigger, line-noise filter, and save derivatives/clean."""
    logger.info("=== Line-noise filtering %s (%s) ===", subject, task)

    # Read the raw recording from the INPUT layout.
    src_layout = BIDSLayout(raw_root, derivatives=False)
    raw = raw_from_layout(
        src_layout, subject=subject, preload=True, extension=".edf"
    )

    # Build the list of EEG / marker channels to drop.
    eeg_electrode_list = load_eeg_chs(subject, eeg_dir)
    eeg_electrode_list = [c for c in eeg_electrode_list if isinstance(c, str)]
    if cfg.should_drop_trigger(subject):
        eeg_electrode_list.append("Trigger")
    # Only drop channels that actually exist (avoids wiping everything if a name
    # is off; original relied on names matching exactly).  Warn about any
    # requested-but-absent channel so a stale EEG list still surfaces.
    to_drop = [c for c in eeg_electrode_list if c in raw.ch_names]
    missing = [c for c in eeg_electrode_list if c not in raw.ch_names]
    if missing:
        logger.warning(
            "Requested drop channels not found in %s (skipped): %s",
            subject, missing,
        )
    logger.info("Dropping %d EEG/marker channels: %s", len(to_drop), to_drop)
    raw.drop_channels(to_drop)

    # Line-noise filtering (same parameters as batch_preproc.py L159-163).
    pp = cfg.table("preproc")
    line_freqs = pp["line_freqs"]
    notch_widths = pp["line_notch_widths"]
    mt_bandwidth = pp["mt_bandwidth"]
    line_filter(
        raw, mt_bandwidth=mt_bandwidth, n_jobs=-1, copy=False, verbose=10,
        filter_length="700ms", freqs=[line_freqs[0]], notch_widths=notch_widths,
    )
    line_filter(
        raw, mt_bandwidth=mt_bandwidth, n_jobs=-1, copy=False, verbose=10,
        filter_length="20s", freqs=line_freqs, notch_widths=notch_widths,
    )

    # Per-dataset special case: some subjects have trailing empty data.
    if cfg.needs_crop(subject):
        raw = crop_empty_data(raw)

    # Save the clean derivative into the OUTPUT base.
    ensure_dir(path.join(bids_root, "derivatives", "clean"))
    out_layout = BIDSLayout(bids_root, derivatives=False)
    save_derivative(raw, out_layout, "clean", True)

    # Strip BAD/EDGE boundary rows from the clean events.tsv files.
    update_tsv(subject, channels_tsv_dir(bids_root, subject), task)
    logger.info("Line-noise filtering complete for %s", subject)


def outlierchs(subject: str, task: str, bids_root: str, cfg) -> None:
    """Mark outlier channels (bad/outlier) in the clean channels.tsv."""
    logger.info("=== Outlier channel marking %s (%s) ===", subject, task)

    layout = BIDSLayout(bids_root, derivatives=True)
    raw = raw_from_layout(
        layout.derivatives["derivatives/clean"],
        subject=subject,
        desc="clean",
        extension=".edf",
        preload=True,
    )

    # Re-entrancy guard: skip if outliers already marked.
    if detect_outlier(subject, channels_tsv_dir(bids_root, subject), task) == 1:
        raise ValueError(
            f"Outlier channels for {subject} are already marked. Skipping. "
            "To re-run, first reset all channels to good / n/a in the clean "
            "channels.tsv."
        )

    outlier = cfg.table("preproc")["outlier"]
    raw.info["bads"] = channel_outlier_marker(raw, *outlier)
    logger.info("Outliers for %s: %s", subject, raw.info["bads"])
    update(raw, layout, "outlier")
    logger.info("Outlier marking complete for %s", subject)


def main(
    subject: str,
    task: str = None,
    bids_root: str = None,
    raw_root: str = None,
    eeg_dir: str = None,
    config: str = None,
    **_,
) -> None:
    cfg = _config.load(config)
    task = task or cfg.input_task
    bids_root = bids_root or cfg.bids_root
    raw_root = raw_root or bids_root
    eeg_dir = eeg_dir or cfg.eeg_dir
    linenoise(subject, task, raw_root, bids_root, eeg_dir, cfg)
    outlierchs(subject, task, bids_root, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject", type=str, required=True,
        help="BIDS subject id, e.g. D0100",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task tag; default = dataset.toml [task].input_task",
    )
    parser.add_argument(
        "--bids_root", type=str, default=None,
        help="Output BIDS base; default = dataset.toml [paths].bids_root",
    )
    parser.add_argument(
        "--raw_root", type=str, default=None,
        help="Input BIDS base to read raw from (default = --bids_root)",
    )
    parser.add_argument(
        "--eeg_dir", type=str, default=None,
        help="Per-subject <S>_eeg_chans.csv dir; default = dataset.toml [paths].eeg_dir",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dataset.toml (default: $LEXPIPE_DATASET_CONFIG or upward search)",
    )
    args = parser.parse_args()
    main(**vars(args))
