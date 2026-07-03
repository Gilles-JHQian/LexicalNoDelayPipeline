"""lexical_pipeline — reusable iEEG preprocessing/analysis core.

Shared, dataset-agnostic engine for the Lexical paradigm family (LexicalDecRep
NoDelay, the upcoming Uniqueness Point experiment, ...). Task-specific drivers,
stimulus metadata, subject lists and sbatch launchers live in each dataset's own
repo and import from here.

Subpackages:
    preprocess    — denoise / muscle / bipolar / parcellation / epoching engine
    lexicon       — uniqueness-point (UP/DP) computation + landmark-aligned epoching
    stats         — permutation-cluster HGA machinery + result packaging
    decoding      — task-agnostic decoders + dataset-prep helpers
    decomposition — NMF / Semi-NMF / TCA / dPCA math
"""

__version__ = "0.0.1"
