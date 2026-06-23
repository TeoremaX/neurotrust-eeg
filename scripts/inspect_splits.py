from __future__ import annotations

import argparse

import pandas as pd

from neurotrust_eeg.config import load_config
from neurotrust_eeg.data import assert_no_subject_leakage, make_subject_split


def print_split_summary(name: str, split: pd.DataFrame) -> None:
    print(f"\n{name.upper()}")
    print("-" * 40)
    print(f"Rows: {len(split):,}")

    if len(split) == 0:
        print("Empty split.")
        return

    print(f"Subjects: {sorted(split['subject'].unique())}")
    print("Labels:")
    print(split["label"].value_counts().sort_index())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    index = pd.read_csv(config["data"]["index_csv"])

    splits = make_subject_split(
        index=index,
        val_subjects=config["data"]["val_subjects"],
        test_subjects=config["data"]["test_subjects"],
    )

    assert_no_subject_leakage(splits)

    print("OK: train/validation/test subjects are disjoint.")

    for name, split in splits.items():
        print_split_summary(name, split)


if __name__ == "__main__":
    main()