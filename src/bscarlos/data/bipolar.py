"""Bipolar channel derivation from raw 10-20 electrodes.

Extracted from notebooks/00_etl-2.ipynb cell 49.
"""

from __future__ import annotations

import pandas as pd

BIPOLAR_PAIRS: list[tuple[str, str, str]] = [
    ("Fp1-F7", "EEG Fp1", "EEG F7"),
    ("F7-T3", "EEG F7", "EEG T3"),
    ("T3-T5", "EEG T3", "EEG T5"),
    ("Fp2-F8", "EEG Fp2", "EEG F8"),
    ("F8-T4", "EEG F8", "EEG T4"),
    ("T4-T6", "EEG T4", "EEG T6"),
]


def create_bipolar_channels(df: pd.DataFrame) -> pd.DataFrame:
    bipolar_df = pd.DataFrame(index=df.index)
    for name, ch1, ch2 in BIPOLAR_PAIRS:
        ch1 = ch1.strip()
        ch2 = ch2.strip()
        if ch1 in df.columns and ch2 in df.columns:
            bipolar_df[name] = df[ch1] - df[ch2]
        else:
            print(f"Warning: Channel(s) missing for pair {name}")
    return bipolar_df
