"""Parcellation utilities shared by decoding-prep scripts.

`process_parc` collapses the a2009s gross_label rows into the ROIs used
downstream. Two schemes:

- 'fine' (default; current behavior): SMC = PrG + PoG + Subcentral, and
  INS is split into AIC / PIC by coordinate + sub-label.
- 'coarse_lobe': lobe-level partition. Temporal labels stay individual
  (STG/HG/STS/MTG/ITG/TP), the rest collapse into Frontal / Parietal /
  Occipital / SMC / INS / Cingulate, with non-cortical and MTL labels
  routed to 'DROP'.
"""

from __future__ import annotations

import pandas as pd


COARSE_LOBE_MAP = {
    'SMC': {'PrG', 'PoG', 'Subcentral', 'Paracentral', 'Central'},
    'Frontal': {
        'SFG', 'SFGs', 'MFG', 'MFGs', 'IFG', 'IFGs',
        'OFC', 'OFCs', 'GRect', 'GSubcallosal',
        'Frontomargin', 'TransvFrontopol',
    },
    'Parietal': {'SPL', 'AG', 'SMG', 'IPL', 'PCun', 'PCuns'},
    'Occipital': {
        'sOccG', 'mOccG', 'iOccG', 'iOccGs',
        'Cun', 'OPC', 'Calcarine', 'OccS',
        'FuG', 'FuGs', 'LinG', 'LinGs',
        'CollatAnt', 'CollatPost',
    },
    'Cingulate': {'CG', 'CGs'},
    # INS is left as 'INS' under coarse_lobe (no AIC/PIC split).
    # Temporal labels untouched: STG, HG, STS, MTG, ITG, TP.
}

COARSE_DROP = {
    'Hipp', 'Amyg', 'PhG',
    'Unknown', 'Intersection', 'UNMAPPED',
    'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter',
    'WM', 'CC', 'CC_Anterior', 'CC_Posterior',
    'LatV', 'InfLatV', 'CP',
    'BrainStem', 'Cb',
    'Sylvian',
    'Thal', 'Put', 'Caud', 'Pallidum', 'VDC',
    'Left-Thalamus', 'Right-Thalamus', 'Left-Pallidum',
}


def _apply_aic_pic_split(parc_: pd.DataFrame, y_threshold: float) -> pd.DataFrame:
    aic_conditions = (
        (parc_['roi'] == 'INS') &
        (
            (parc_['label'].str.contains('G_insular_short', na=False)) |
            (parc_['label'].str.contains('S_circular_insula_ant', na=False)) |
            ((parc_['label'].str.contains('S_circular_insula_sup', na=False)) &
             (parc_['y'] > y_threshold)) |
            ((parc_['label'].str.contains('S_circular_insula_inf', na=False)) &
             (parc_['y'] > y_threshold))
        )
    )
    pic_conditions = (
        (parc_['roi'] == 'INS') &
        (
            (parc_['label'].str.contains('G_Ins_lg_and_S_cent_ins', na=False)) |
            ((parc_['label'].str.contains('S_circular_insula_sup', na=False)) &
             (parc_['y'] <= y_threshold)) |
            ((parc_['label'].str.contains('S_circular_insula_inf', na=False)) &
             (parc_['y'] <= y_threshold))
        )
    )
    parc_.loc[aic_conditions, 'roi'] = 'AIC'
    parc_.loc[pic_conditions, 'roi'] = 'PIC'
    return parc_


def hemi_groups(roi: str, hemi_mode: str = 'split') -> list[str]:
    """Hemisphere groups to build for one ROI.

    Returns the list of hemi tokens to iterate when preparing decoding
    datasets. The 'all' ROI is always a single 'all' group (every channel).
    For a real ROI the mode controls left/right handling:

    - 'split' (default): ['L', 'R']         -> per-hemisphere, subject suffix l/r
                                               (e.g. 'STGl', 'STGr')
    - 'merge':           ['both']           -> pool both hemispheres into one
                                               bilateral dataset, no l/r suffix
                                               (e.g. 'STG')
    - 'both':            ['L', 'R', 'both'] -> per-hemisphere AND the merged
                                               bilateral dataset

    The 'both' hemi token means "select channels from this ROI in either
    hemisphere"; callers should branch on it when picking channels.
    """
    if roi == 'all':
        return ['all']
    if hemi_mode == 'split':
        return ['L', 'R']
    if hemi_mode == 'merge':
        return ['both']
    if hemi_mode == 'both':
        return ['L', 'R', 'both']
    raise ValueError(
        f"Unknown hemi_mode: {hemi_mode!r}; expected 'split', 'merge', or 'both'"
    )


def hemi_label(roi: str, hemi: str) -> str:
    """BIDS subject token for a (roi, hemi) group.

    Single hemispheres get an l/r suffix ('STGl', 'STGr'); the merged ('both')
    and whole-brain ('all') groups carry no suffix ('STG', 'all').
    """
    if hemi in ('L', 'R'):
        return f"{roi}{hemi.lower()}"
    return f"{roi}"


def process_parc(
    parc: pd.DataFrame,
    y_threshold: float = 0.0,
    scheme: str = 'fine',
) -> pd.DataFrame:
    parc_ = parc.copy()
    if scheme == 'fine':
        parc_['roi'] = parc_['roi'].replace(
            {'PrG': 'SMC', 'PoG': 'SMC', 'Subcentral': 'SMC'}
        )
        parc_ = _apply_aic_pic_split(parc_, y_threshold)
        return parc_
    if scheme == 'coarse_lobe':
        new_roi = parc_['roi'].copy()
        for target, members in COARSE_LOBE_MAP.items():
            new_roi[new_roi.isin(members)] = target
        new_roi[new_roi.isin(COARSE_DROP)] = 'DROP'
        parc_['roi'] = new_roi
        return parc_
    raise ValueError(f"Unknown scheme: {scheme!r}; expected 'fine' or 'coarse_lobe'")
