from __future__ import annotations

import argparse
from pathlib import Path

from neurotrust_eeg.config import load_config
from neurotrust_eeg.data import WindowSpec, build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite index CSV if it already exists.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    data_config = config["data"]
    index_csv = Path(data_config["index_csv"])

    if index_csv.exists() and not args.force:
        print(f"Index already exists: {index_csv}")
        print("Use --force to rebuild it.")
        return

    spec = WindowSpec(
        window_seconds=float(data_config["window_seconds"]),
        stride_seconds=float(data_config["stride_seconds"]),
        sample_rate=int(data_config["sample_rate"]),
        min_seizure_overlap=float(data_config["min_seizure_overlap"]),
    )

    index = build_index(
        root=data_config["root"],
        spec=spec,
        save_csv=index_csv,
    )

    print(f"Saved index to: {index_csv}")
    print(f"Total windows: {len(index):,}")

    if len(index) > 0:
        print()
        print("Windows by label:")
        print(index["label"].value_counts().sort_index())

        print()
        print("Subjects:")
        print(sorted(index["subject"].unique()))


if __name__ == "__main__":
    main()