from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    sample_rate: int = 256
    min_seizure_overlap: float = 0.25


def overlap_fraction(
    window_start: float,
    window_end: float,
    event_start: float,
    event_end: float,
) -> float:
    """Return the fraction of a window that overlaps with an event."""

    overlap = max(0.0, min(window_end, event_end) - max(window_start, event_start))
    window_duration = max(1e-8, window_end - window_start)

    return overlap / window_duration


def parse_summary_file(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Parse a CHB-MIT summary file.

    Returns:
        Mapping from EDF filename to seizure intervals in seconds.
    """

    path = Path(path)
    text = path.read_text(errors="ignore", encoding="utf-8")
    lines = text.splitlines()

    annotations: dict[str, list[tuple[float, float]]] = {}
    current_file: str | None = None
    pending_starts: list[float] = []

    for line in lines:
        file_match = re.search(r"File Name:\s*(\S+)", line)

        if file_match:
            current_file = file_match.group(1)
            annotations.setdefault(current_file, [])
            pending_starts = []
            continue

        if current_file is None:
            continue

        start_match = re.search(
            r"Seizure(?: \d+)? Start Time:\s*([\d.]+)\s*seconds",
            line,
        )
        end_match = re.search(
            r"Seizure(?: \d+)? End Time:\s*([\d.]+)\s*seconds",
            line,
        )

        if start_match:
            pending_starts.append(float(start_match.group(1)))

        elif end_match and pending_starts:
            start = pending_starts.pop(0)
            end = float(end_match.group(1))
            annotations[current_file].append((start, end))

    return annotations


def build_index(
    root: str | Path,
    spec: WindowSpec,
    save_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Build a window-level metadata index for CHB-MIT.

    This function reads EDF headers and summary files, but not the full EEG data.
    """

    root = Path(root)
    rows: list[dict] = []

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    subject_dirs = sorted(path for path in root.glob("chb*") if path.is_dir())

    if not subject_dirs:
        raise RuntimeError(f"No CHB-MIT subject folders found in: {root}")

    for subject_dir in subject_dirs:
        annotations: dict[str, list[tuple[float, float]]] = {}

        for summary_file in subject_dir.glob("*summary.txt"):
            annotations.update(parse_summary_file(summary_file))

        edf_files = sorted(subject_dir.glob("*.edf"))

        for edf_path in edf_files:
            try:
                import mne

                raw = mne.io.read_raw_edf(
                    edf_path,
                    preload=False,
                    verbose="ERROR",
                )

                duration_seconds = raw.n_times / raw.info["sfreq"]
                original_sample_rate = raw.info["sfreq"]
                channels = ",".join(raw.ch_names)

            except Exception as exc:
                print(f"WARNING: Could not read EDF header: {edf_path} ({exc})")
                continue

            seizure_intervals = annotations.get(edf_path.name, [])

            current_start = 0.0

            while current_start + spec.window_seconds <= duration_seconds:
                current_end = current_start + spec.window_seconds

                label = int(
                    any(
                        overlap_fraction(
                            current_start,
                            current_end,
                            seizure_start,
                            seizure_end,
                        )
                        >= spec.min_seizure_overlap
                        for seizure_start, seizure_end in seizure_intervals
                    )
                )

                rows.append(
                    {
                        "subject": subject_dir.name,
                        "file": str(edf_path),
                        "file_name": edf_path.name,
                        "start_sec": current_start,
                        "end_sec": current_end,
                        "label": label,
                        "sfreq_original": original_sample_rate,
                        "channels": channels,
                    }
                )

                current_start += spec.stride_seconds

    index = pd.DataFrame(rows)

    if save_csv is not None:
        save_path = Path(save_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(save_path, index=False)

    return index


def make_subject_split(
    index: pd.DataFrame,
    val_subjects: list[str],
    test_subjects: list[str],
) -> dict[str, pd.DataFrame]:
    """Split metadata by subject to avoid data leakage."""

    val_subjects_set = set(val_subjects)
    test_subjects_set = set(test_subjects)

    if val_subjects_set & test_subjects_set:
        raise ValueError("Validation and test subjects overlap.")

    train = index[
        ~index["subject"].isin(val_subjects_set | test_subjects_set)
    ].copy()

    validation = index[index["subject"].isin(val_subjects_set)].copy()
    test = index[index["subject"].isin(test_subjects_set)].copy()

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def assert_no_subject_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise an error if any subject appears in more than one split."""

    split_subjects = {
        split_name: set(split_df["subject"].unique())
        for split_name, split_df in splits.items()
    }

    names = list(split_subjects)

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            overlap = split_subjects[name_a] & split_subjects[name_b]

            if overlap:
                raise AssertionError(
                    f"Subject leakage between {name_a} and {name_b}: {sorted(overlap)}"
                )