"""Per-dataset configuration loader.

Single source of truth for everything the pipeline used to hardcode (paths, task
names, condition/phase vocabulary, per-dataset preprocessing tables). Each dataset
repo ships a ``dataset.toml``; the CLIs read it as their argparse defaults, and a
``config export`` step renders the same values into ``config.sh`` for sbatch.

Resolution order for the config file:
  1. explicit ``--config <path>`` (passed to ``load``),
  2. ``$LEXPIPE_DATASET_CONFIG`` environment variable,
  3. nearest ``dataset.toml`` found walking up from the current directory.

Relative paths inside the toml resolve against the toml's own directory, so the
config is independent of where the code lives or the current working directory.
"""
from __future__ import annotations

import os
import tomllib
from typing import Any


def find_config(explicit: str | None = None) -> str:
    """Locate the dataset.toml (see module docstring for order)."""
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("LEXPIPE_DATASET_CONFIG")
    if env:
        return os.path.abspath(env)
    d = os.getcwd()
    while True:
        cand = os.path.join(d, "dataset.toml")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "dataset.toml not found. Pass --config <path> or set "
        "$LEXPIPE_DATASET_CONFIG, or run from inside a dataset repo."
    )


class DatasetConfig:
    """Parsed dataset.toml with typed accessors and path resolution."""

    def __init__(self, data: dict[str, Any], root: str, source: str):
        self._d = data
        self.root = root      # directory containing dataset.toml
        self.source = source  # full path to dataset.toml (provenance)

    # -- raw access -----------------------------------------------------------
    def table(self, name: str) -> dict[str, Any]:
        return self._d.get(name, {})

    def _resolve(self, value: str) -> str:
        """Absolute path: as-is if absolute, else relative to the toml's dir."""
        return value if os.path.isabs(value) else os.path.normpath(
            os.path.join(self.root, value)
        )

    def path(self, table: str, key: str, default: str | None = None) -> str | None:
        val = self._d.get(table, {}).get(key, default)
        return None if val is None else self._resolve(val)

    # -- [paths] --------------------------------------------------------------
    @property
    def bids_root(self) -> str:
        return self.path("paths", "bids_root")

    @property
    def raw_root(self) -> str:
        return self.path("paths", "raw_root") or self.bids_root

    @property
    def recon_dir(self) -> str:
        return self.path("paths", "recon_dir")

    @property
    def result_root(self) -> str:
        return self.path("paths", "result_root")

    @property
    def eeg_dir(self) -> str:
        return self.path("paths", "eeg_dir")

    @property
    def muscle_dir(self) -> str:
        return self.path("paths", "muscle_dir")

    # -- [task] ---------------------------------------------------------------
    @property
    def input_task(self) -> str:
        """BIDS acquisition task (raw filenames / --task)."""
        return self._d["task"]["input_task"]

    @property
    def output_task(self) -> str:
        """Derivative task tag written into output BIDSPaths (load-bearing)."""
        return self._d["task"]["output_task"]

    # -- [preproc] helpers ----------------------------------------------------
    def should_drop_trigger(self, subject: str) -> bool:
        """Whether to force-drop a 'Trigger' channel for this subject."""
        return subject not in set(self.table("preproc").get("trigger_nodrop", []))

    def needs_crop(self, subject: str) -> bool:
        """Whether this subject needs crop_empty_data (trailing empty recording)."""
        return subject in set(self.table("preproc").get("crop_empty", []))

    def recon_id(self, subject: str) -> str:
        """Map a BIDS subject id to its FreeSurfer recon id (e.g. D0019 -> D19)."""
        fmt = self.table("preproc").get("recon_id_format", "D{n}")
        return fmt.format(n=int(subject[1:]))


def load(path: str | None = None) -> DatasetConfig:
    """Load the dataset config (see module docstring)."""
    p = find_config(path)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return DatasetConfig(data, os.path.dirname(p), p)
