import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG window tensors."""

    def __init__(self, data: np.ndarray | torch.Tensor):
        if isinstance(data, torch.Tensor):
            self.data = data.to(dtype=torch.float32)
        else:
            self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], 0  # Dummy label for VAE compatibility


def normalize_eeg_data(eeg_array: np.ndarray) -> np.ndarray:
    """Global min-max normalization to [0, 1] (notebooks/03_phase2_vae.ipynb cell 7).

    Required before windowing/training: the VAE decoder outputs via sigmoid
    (bounded to [0, 1]), so the reconstruction target must be in the same range
    or the loss is meaningless. Confirmed empirically: unnormalized real-amplitude
    EEG data produces reconstruction losses on the order of 1e15.
    """
    eeg_min = eeg_array.min()
    eeg_max = eeg_array.max()
    return (eeg_array - eeg_min) / (eeg_max - eeg_min)


def reshape_to_windows(eeg_array: np.ndarray, window_size: int = 512, n_channels: int = 6) -> np.ndarray:
    """Reshapes continuous EEG array (samples, n_channels) into windows (n_windows, window_size * n_channels)."""
    if eeg_array.ndim != 2:
        raise ValueError(f"Expected 2D EEG array of shape (samples, n_channels), got shape {eeg_array.shape}")
    num_windows = eeg_array.shape[0] // window_size
    trimmed = eeg_array[: num_windows * window_size]
    reshaped = trimmed.reshape(num_windows, window_size * n_channels)
    return reshaped
