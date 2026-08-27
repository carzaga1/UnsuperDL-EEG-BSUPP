"""Ground-truth label construction from annotator export files.

Extracted from notebooks/00_etl-2.ipynb cells 57-65. Different patients' real
annotation files use "Burst starts" (plural, e.g. SE008) and "Burst start"
(singular, e.g. SE021) inconsistently, so both are treated as the start marker.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BURST_START_LABELS = {"Burst starts", "Burst start"}
BURST_END_LABEL = "Burst end"


def parse_annotations(path: Path) -> pd.DataFrame:
    annotations = pd.read_csv(path)
    annotations["Onset"] = pd.to_datetime(annotations["Onset"], format="%Y-%m-%dT%H:%M:%S.%f")
    return annotations


def build_ground_truth_labels(timestamps: pd.DatetimeIndex, annotations: pd.DataFrame) -> pd.Series:
    ground_truth = pd.Series(0, index=timestamps, name="ground_truth")

    for index, row in annotations.iterrows():
        if row["Annotation"] not in BURST_START_LABELS:
            continue

        start_time = row["Onset"]
        end_matches = annotations.loc[
            (annotations["Annotation"] == BURST_END_LABEL) & (annotations.index > index), "Onset"
        ]
        if end_matches.empty:
            print(f"Warning: No matching 'Burst end' annotation found for burst start at {start_time}")
            continue

        end_time = end_matches.iloc[0]
        ground_truth.loc[(ground_truth.index >= start_time) & (ground_truth.index <= end_time)] = 1

    return ground_truth
