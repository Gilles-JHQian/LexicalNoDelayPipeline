"""Uniqueness-point (UP) aligned epoching utilities.

The stimuli are spoken words/nonwords with forced-alignment phoneme timing in
``<annotations_dir>/<token>_phones.txt``. Each token has a uniqueness point (UP,
words) / deviation point (DP, nonwords) -- a 1-indexed phoneme position stored in
the stimulus-properties CSV (``uniqueness_point`` column).

This module converts that phoneme index into a per-trial time offset (seconds
from auditory-stimulus onset) and cuts ``Auditory_stim`` epochs re-aligned to
that moment -- i.e. time-locked to when the word becomes lexically identifiable.

Dataset-agnostic: all stimulus paths are passed in by the caller (the per-dataset
driver supplies them from ``dataset.toml``). Mirrors the per-trial event-shifting
used for the Delay phase in the epoching engine, but shifts by the UP-phoneme
onset instead of the stimulus duration.
"""
import csv
import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)


def read_phone_timings(token, ann_dir):
    """Read forced-alignment phoneme timings for one token.

    File format: tab-separated ``start  end  PHONE`` per line, times in seconds
    relative to stimulus onset. Returns a list of ``(phone, start, end)`` tuples
    in order, or ``None`` if the annotation file is absent/empty.
    """
    import os
    path = os.path.join(ann_dir, f"{token}_phones.txt")
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out.append((parts[2], float(parts[0]), float(parts[1])))
    return out or None


def load_up_offsets(stim_csv, ann_dir):
    """Build ``dict[token -> offset_seconds]`` for UP-aligned epoching.

    ``offset`` is the onset time of the UP-th phoneme (relative to stimulus
    onset). When ``UP == n_phonemes + 1`` (word unique only at offset) the
    offset is the end time of the last phoneme (the word offset). Words use UP,
    nonwords use DP -- both are 1-indexed phoneme positions in the same column.
    """
    offsets = {}
    with open(stim_csv, newline="") as f:
        for row in csv.DictReader(f):
            token = row.get("content")
            up_raw = row.get("uniqueness_point", "")
            if not token or up_raw == "":
                continue
            phones = read_phone_timings(token, ann_dir)
            if not phones:
                logger.warning(f"UP offsets: no phone annotation for {token!r}")
                continue
            up = int(up_raw)
            if up <= len(phones):
                offsets[token] = phones[up - 1][1]      # UP phoneme onset
            else:
                offsets[token] = phones[-1][2]          # word offset (last end)
    return offsets


def _token_from_description(desc):
    """Extract the stimulus token from a hierarchical annotation description.

    e.g. ``'Auditory_stim/Yes_No/Word/humor/CORRECT'`` -> ``'humor'`` (index 3).
    """
    parts = desc.split("/")
    return parts[3] if len(parts) > 3 else None


def cut_up_aligned_epochs(raw, events, event_id, condition, tmin, tmax,
                          up_offsets, logger=logger):
    """Cut ``Auditory_stim`` epochs re-aligned to each trial's UP phoneme onset.

    Mirrors the Delay-phase event shifting in the epoching engine but shifts each
    event by its token's UP offset (from ``up_offsets``) instead of the stimulus
    duration. Trials whose token is missing from ``up_offsets`` are dropped with a
    warning (expected: none).

    Parameters
    ----------
    raw : mne.io.Raw
    events, event_id : output of ``mne.events_from_annotations(raw)``
    condition : str   one of the raw condition tags ('Yes_No', 'Repeat', ':=:')
    tmin, tmax : float   epoch window (the ``+/-0.5`` padding matches Delay)
    up_offsets : dict[str, float]   token -> offset seconds (see load_up_offsets)

    Returns
    -------
    mne.Epochs | None
        UP-aligned epochs (original event names; caller runs ``pick_event_name``
        and saving), or ``None`` if nothing alignable.
    """
    src_events = [e for e in event_id.keys() if f"Auditory_stim/{condition}" in e]
    if not src_events:
        logger.warning(f"UP: no Auditory_stim events for condition={condition!r}. "
                       "Skipping.")
        return None

    sfreq = raw.info["sfreq"]
    src_set = set(src_events)
    code_set = set(event_id[n] for n in src_events if n in event_id)

    # Auditory_stim events (sample-ordered) and matching annotations (onset-ordered).
    aud_events_arr = events[np.isin(events[:, 2], list(code_set))].copy()
    aud_anns = sorted(
        [ann for ann in raw.annotations if ann["description"] in src_set],
        key=lambda a: a["onset"],
    )
    if len(aud_events_arr) != len(aud_anns):
        logger.warning(
            f"UP: events ({len(aud_events_arr)}) and annotations "
            f"({len(aud_anns)}) count mismatch for Auditory_stim/{condition}. "
            "Skipping."
        )
        return None

    # Shift each event sample by its token's UP offset; drop unmatched tokens.
    offset_events = aud_events_arr.copy()
    keep = np.ones(len(aud_anns), dtype=bool)
    missing = []
    for i, ann in enumerate(aud_anns):
        token = _token_from_description(ann["description"])
        off = up_offsets.get(token)
        if off is None:
            keep[i] = False
            missing.append(token)
            continue
        offset_events[i, 0] += int(round(off * sfreq))

    if missing:
        logger.warning(
            f"UP: dropping {int((~keep).sum())} trials with no UP offset "
            f"(tokens: {sorted(set(missing))})"
        )
    offset_events = offset_events[keep]
    if len(offset_events) == 0:
        logger.warning(f"UP: no alignable trials for condition={condition!r}.")
        return None
    offset_events = offset_events[offset_events[:, 0].argsort()]

    # Restrict event_id to codes actually present (avoids MNE zero-event error).
    present = set(offset_events[:, 2])
    code_to_name = {event_id[n]: n for n in src_events if n in event_id}
    offset_event_id = {code_to_name[c]: c for c in present if c in code_to_name}

    return mne.Epochs(
        raw,
        events=offset_events,
        event_id=offset_event_id,
        tmin=tmin - 0.5,
        tmax=tmax + 0.5,
        preload=True,
        baseline=None,
    )
