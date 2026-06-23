from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/chbmit",
        help="Directory where CHB-MIT will be downloaded.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wfdb
    except ImportError as exc:
        raise SystemExit("wfdb is required. Install it with: pip install wfdb") from exc

    print(f"Downloading CHB-MIT to: {output_dir.resolve()}")
    print("This can take a while because the dataset is large.")

    wfdb.dl_database(
        db_dir="chbmit",
        dl_dir=str(output_dir),
    )

    print("Download completed.")


if __name__ == "__main__":
    main()