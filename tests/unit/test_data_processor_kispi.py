import pandas as pd

from bscarlos.data_processor_kispi import PatientDataProcessor
from bscarlos.testing.synthetic_data import (
    make_synthetic_data_attributes,
    make_synthetic_processed_parquet,
)


def _write_patient_parquet(tmp_path, patient_id, n_windows=2000, seed=0):
    df = make_synthetic_processed_parquet(channel_set="6ch", n_windows=n_windows, seed=seed)
    path = tmp_path / f"{patient_id}_filt_merged.parquet"
    df.to_parquet(path)
    return path


def test_create_patient_data_dict(tmp_path):
    patient_id = "SE008_a1"
    parquet_path = _write_patient_parquet(tmp_path, patient_id)
    data_attributes = make_synthetic_data_attributes([patient_id], [256])

    processor = PatientDataProcessor()
    patient_data = processor.create_patient_data_dict([parquet_path], data_attributes)

    assert patient_id in patient_data
    assert patient_data[patient_id]["fs"] == 256
    assert isinstance(patient_data[patient_id]["data"], pd.DataFrame)
    assert "ground_truth" in patient_data[patient_id]["data"].columns


def test_apply_timedelta_index(tmp_path):
    patient_id = "SE008_a1"
    parquet_path = _write_patient_parquet(tmp_path, patient_id, n_windows=100)
    data_attributes = make_synthetic_data_attributes([patient_id], [200])

    processor = PatientDataProcessor()
    patient_data = processor.create_patient_data_dict([parquet_path], data_attributes)
    patient_data = processor.apply_timedelta_index(patient_data)

    index = patient_data[patient_id]["data"].index
    assert isinstance(index, pd.TimedeltaIndex)
    assert index[0] == pd.Timedelta(0)
    assert index[1] == pd.Timedelta("1s") / 200


def test_apply_bandpass_filter_preserves_shape_and_ground_truth(tmp_path):
    patient_id = "SE008_a1"
    parquet_path = _write_patient_parquet(tmp_path, patient_id)
    data_attributes = make_synthetic_data_attributes([patient_id], [256])

    processor = PatientDataProcessor()
    patient_data = processor.create_patient_data_dict([parquet_path], data_attributes)
    original = patient_data[patient_id]["data"].copy()

    filtered_data = processor.apply_bandpass_filter(patient_data)
    filtered_df = filtered_data[patient_id]["data"]

    assert filtered_df.shape == original.shape
    assert list(filtered_df.columns) == list(original.columns)
    pd.testing.assert_series_equal(filtered_df["ground_truth"], original["ground_truth"])


def test_split_data_for_5fold_cv(tmp_path):
    patient_id = "SE008_a1"
    parquet_path = _write_patient_parquet(tmp_path, patient_id, n_windows=100)
    data_attributes = make_synthetic_data_attributes([patient_id], [256])

    processor = PatientDataProcessor()
    patient_data = processor.create_patient_data_dict([parquet_path], data_attributes)
    patient_data = processor.split_data_for_5fold_cv(patient_data, n_splits=5)

    assert len(patient_data[patient_id]["train_folds"]) == 5
    assert len(patient_data[patient_id]["test_folds"]) == 5
    for train_fold, test_fold in zip(
        patient_data[patient_id]["train_folds"], patient_data[patient_id]["test_folds"]
    ):
        assert len(train_fold) + len(test_fold) == 100


def test_remove_ground_truth_column(tmp_path):
    patient_id = "SE008_a1"
    parquet_path = _write_patient_parquet(tmp_path, patient_id, n_windows=100)
    data_attributes = make_synthetic_data_attributes([patient_id], [256])

    processor = PatientDataProcessor()
    patient_data = processor.create_patient_data_dict([parquet_path], data_attributes)
    patient_data = processor.apply_timedelta_index(patient_data)
    patient_data = processor.split_data_for_training(patient_data)
    patient_data = processor.remove_ground_truth_column(patient_data)

    assert "ground_truth" not in patient_data[patient_id]["eeg_training"].columns
    assert "ground_truth" not in patient_data[patient_id]["eeg_testing"].columns
    assert len(patient_data[patient_id]["eeg_training"]) > 0
    assert len(patient_data[patient_id]["eeg_testing"]) > 0
