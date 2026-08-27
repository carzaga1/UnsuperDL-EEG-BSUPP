import numpy as np
import torch

from bscarlos.architectures import EEGDataset, normalize_eeg_data, reshape_to_windows


def test_normalize_eeg_data_scales_to_unit_range():
    data = np.array([[10.0, -5.0], [50.0, 100.0], [0.0, 30.0]])

    normalized = normalize_eeg_data(data)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized.shape == data.shape


def test_reshape_to_windows():
    n_samples = 1200
    n_channels = 6
    window_size = 512

    data = np.random.randn(n_samples, n_channels)
    windows = reshape_to_windows(data, window_size=window_size, n_channels=n_channels)

    expected_windows = n_samples // window_size  # 2
    assert windows.shape == (expected_windows, window_size * n_channels)


def test_eeg_dataset():
    num_windows = 10
    input_dim = 3072
    dummy_data = np.random.randn(num_windows, input_dim).astype(np.float32)

    dataset = EEGDataset(dummy_data)
    assert len(dataset) == num_windows

    sample, label = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (input_dim,)
    assert sample.dtype == torch.float32
    assert label == 0
