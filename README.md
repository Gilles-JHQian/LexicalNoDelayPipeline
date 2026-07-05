# lexical_pipeline

Reusable, dataset-agnostic iEEG preprocessing/analysis core for the **Lexical
paradigm family** (LexicalDecRep NoDelay, the upcoming Uniqueness Point
experiment, …). Task-specific drivers, stimulus metadata, subject lists and
sbatch launchers live in each dataset's own repo (e.g. `lexical_nodelay`) and
import from here.

## Install (offline, on the HPC conda env)

Runtime dependencies (numpy/pandas/scipy/scikit-learn/mne/mne-bids/h5py/tqdm/
tensorly + CoganLab `ieeg`) are already provided by the per-dataset conda env,
so install editable with **no dependency resolution / no build isolation**:

```bash
conda activate Lexical_NoDelay
pip install -e /hpc/home/jq81/repos/LexicalNoDelayPipeline --no-deps --no-build-isolation
```

## Layout

```
src/lexical_pipeline/
  preprocess/    denoise / apply_muscle / apply_eeg / save_bipolar_derivative / parcellation / epoching engine
  lexicon/       uniqueness-point (UP/DP) computation + landmark-aligned epoching
  stats/         permutation-cluster HGA machinery + result packaging
  decoding/      decoder / cross_decoder / parc_utils + decoding engines
  decomposition/ NMF / Semi-NMF / TCA / dPCA math
  data/          packaged atlas resources (FreeSurferColorLUT.txt, a2009s.csv)
```

## Status

Migrated out of `BIDS-1.0_LexicalDecRepNoDelay/BIDS/code` incrementally. See that
repo's `REFACTOR_PROPOSAL.md` / `REFACTOR_MANIFEST.md` for the file-by-file plan.

Phase 1 (agnostic leaf modules): `decoding.decoder`, `decoding.cross_decoder`,
`decoding.parc_utils`, `preprocess.save_bipolar_derivative`.
