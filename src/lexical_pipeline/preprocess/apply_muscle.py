"""Apply manually-marked muscle channels to the clean channels.tsv (contract B).

The companion to ``denoise.py``.  Muscle channels are identified outside this
pipeline (the sister Muscle GUI tool / by inspection) and exported as one CSV
per subject.  This script reads those CSVs and writes ``status=bad,
status_description=muscle`` into every run's clean channels.tsv.  Idempotent.

``--subject all`` walks every ``*_muscle_chans.csv`` found in ``--muscle_dir``.
Dataset-specific values (bids_root/task/muscle_dir) come from ``dataset.toml``
(``--config``); any CLI flag overrides the config value.
"""
import argparse
import glob
import logging
import sys
from os import path

from lexical_pipeline import config as _config
from .preproc_utils import channels_tsv_dir, update_muscle_chs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _subjects_from_muscle_dir(muscle_dir: str) -> list:
    """All subject ids that have a muscle CSV in ``muscle_dir``."""
    files = sorted(glob.glob(path.join(muscle_dir, "*_muscle_chans.csv")))
    return [path.basename(f).replace("_muscle_chans.csv", "") for f in files]


def apply_one(subject: str, task: str, bids_root: str, muscle_dir: str) -> None:
    search_dir = channels_tsv_dir(bids_root, subject)
    if not path.isdir(search_dir):
        logger.warning(
            "No clean derivative for %s at %s; skipping.", subject, search_dir
        )
        return
    chans = update_muscle_chs(subject, search_dir, task, muscle_dir)
    logger.info("Applied %d muscle channel(s) for %s: %s",
                len(chans), subject, chans)


def main(
    subject: str,
    task: str = None,
    bids_root: str = None,
    muscle_dir: str = None,
    config: str = None,
    **_,
) -> None:
    cfg = _config.load(config)
    task = task or cfg.input_task
    bids_root = bids_root or cfg.bids_root
    muscle_dir = muscle_dir or cfg.muscle_dir

    if subject.lower() == "all":
        subjects = _subjects_from_muscle_dir(muscle_dir)
        if not subjects:
            logger.warning("No *_muscle_chans.csv found in %s", muscle_dir)
        logger.info("Applying muscle marks for %d subject(s) from %s",
                    len(subjects), muscle_dir)
    else:
        subjects = [subject]

    for subj in subjects:
        try:
            apply_one(subj, task, bids_root, muscle_dir)
        except (ValueError, FileNotFoundError) as e:
            logger.error("Failed for %s: %s", subj, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject", type=str, required=True,
        help="BIDS subject id, e.g. D0100, or 'all' to walk --muscle_dir",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task tag; default = dataset.toml [task].input_task",
    )
    parser.add_argument(
        "--bids_root", type=str, default=None,
        help="BIDS base whose derivatives/clean channels.tsv get marked "
             "(default = dataset.toml [paths].bids_root)",
    )
    parser.add_argument(
        "--muscle_dir", type=str, default=None,
        help="Directory of muscle-channel result CSVs "
             "(default = dataset.toml [paths].muscle_dir)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dataset.toml (default: $LEXPIPE_DATASET_CONFIG or upward search)",
    )
    args = parser.parse_args()
    main(**vars(args))
