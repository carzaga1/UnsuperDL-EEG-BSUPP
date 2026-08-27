from bscarlos.data.edf_ingest import (
    CHANNELS_TO_DROP,
    drop_nonrelevant_channels,
    find_annotation_edf_files,
    read_edf_raw,
)


def test_drop_nonrelevant_channels(synthetic_edf_file):
    raw = read_edf_raw(synthetic_edf_file)
    raw = drop_nonrelevant_channels(raw)

    assert not any(ch in raw.ch_names for ch in CHANNELS_TO_DROP)
    # The 8 electrodes used for bipolar derivation must survive the drop.
    assert "EEG Fp1" in raw.ch_names
    assert "EEG T6" in raw.ch_names


def test_find_annotation_edf_files(tmp_path):
    (tmp_path / "SE008_annotated1.edf").touch()
    (tmp_path / "SE021_annotated1.edf").touch()
    (tmp_path / "SE008_annotated2.edf").touch()

    a1_files = find_annotation_edf_files(tmp_path, "a1")
    a2_files = find_annotation_edf_files(tmp_path, "a2")

    assert [f.name for f in a1_files] == ["SE008_annotated1.edf", "SE021_annotated1.edf"]
    assert [f.name for f in a2_files] == ["SE008_annotated2.edf"]
