"""EDF ingestion: read raw KISPI recordings and drop non-relevant channels.

Extracted from notebooks/00_etl-2.ipynb cells 0, 3, 5, 7.
"""

from __future__ import annotations

from pathlib import Path

import mne

# Channels dropped before bipolar derivation (notebooks/00_etl-2.ipynb cells 5/7).
CHANNELS_TO_DROP = [
    "EEG F3", "EEG Fz", "EEG F4",
    "EEG C3", "EEG Cz", "EEG C4",
    "EEG P3", "EEG Pz", "EEG P4",
    "EEG O1", "EEG O2", "EEG A1", "EEG A2",
    "NASE", "EKG", "PHO", "EOG", "EMG",
]


def read_edf_raw(path: Path) -> mne.io.Raw:
    return mne.io.read_raw_edf(path, preload=True, verbose=False)


def drop_nonrelevant_channels(raw: mne.io.Raw) -> mne.io.Raw:
    existing_channels = [ch for ch in CHANNELS_TO_DROP if ch in raw.ch_names]
    if existing_channels:
        raw.drop_channels(existing_channels)
    return raw


def find_annotation_edf_files(raw_dir: Path, annotator: str) -> list[Path]:
    annotator_number = annotator.lstrip("a")
    return sorted(Path(raw_dir).glob(f"*annotated{annotator_number}.edf"))
