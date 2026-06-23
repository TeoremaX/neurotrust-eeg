import pandas as pd

from neurotrust_eeg.data import (
    assert_no_subject_leakage,
    make_subject_split,
    overlap_fraction,
    parse_summary_file,
)


def test_overlap_fraction():
    assert overlap_fraction(0.0, 10.0, 2.0, 5.0) == 0.3
    assert overlap_fraction(0.0, 10.0, 20.0, 30.0) == 0.0
    assert overlap_fraction(0.0, 10.0, 0.0, 10.0) == 1.0


def test_parse_summary_file(tmp_path):
    summary = tmp_path / "chb01-summary.txt"
    summary.write_text(
        """
File Name: chb01_03.edf
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
""",
        encoding="utf-8",
    )

    annotations = parse_summary_file(summary)

    assert "chb01_03.edf" in annotations
    assert annotations["chb01_03.edf"] == [(2996.0, 3036.0)]


def test_make_subject_split_without_leakage():
    index = pd.DataFrame(
        {
            "subject": ["chb01", "chb01", "chb02", "chb03"],
            "label": [0, 1, 0, 1],
        }
    )

    splits = make_subject_split(
        index=index,
        val_subjects=["chb02"],
        test_subjects=["chb03"],
    )

    assert set(splits["train"]["subject"]) == {"chb01"}
    assert set(splits["validation"]["subject"]) == {"chb02"}
    assert set(splits["test"]["subject"]) == {"chb03"}

    assert_no_subject_leakage(splits)