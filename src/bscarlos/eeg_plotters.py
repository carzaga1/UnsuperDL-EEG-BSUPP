import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class EEGPlotter:
    def __init__(self, df):
        """
        Initialize the EEGPlotter with the provided DataFrame.

        Parameters:
        df (pd.DataFrame): EEG data with time-indexed rows and columns for each channel.
        """
        self.df = df
        self.t_s = (df.index / pd.Timedelta(1, 's')).to_numpy()  # Convert time index to seconds

    def plot_eeg(self):
        """
        Plot all EEG channels in a single figure with scaling for specific channels.
        """
        plt.figure(figsize=(12, 8))

        for i, col in enumerate(self.df):
            scale = 1
            if str(col) not in {
                'Fp1-F7', 'F7-T3', 'T3-T5', 'Fp1-F3', 'F3-C3', 'C3-P3',
                'P3-O1', 'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4',
                'P4-O2', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'
            }:
                scale = 50

            x = self.df[col].to_numpy()
            plt.plot(self.t_s, scale * x + 100 * i, label=col)

        plt.xlabel('Time [s]')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
        plt.tight_layout()
        plt.show()

    def plot_eeg_multiple(self):
        """
        Plot EEG channels as separate subplots with groundtruth and predicted labels highlighted.
        """
        n_samples = len(self.df.index)
        data = self.df.to_numpy().T
        t_s = (self.df.index - self.df.index[0]).total_seconds()
        five_min = next((i for i, t in enumerate(t_s) if t > 300), n_samples)

        fig, axes = plt.subplots(len(self.df.columns[:-1]), 1, figsize=(12, 8), sharex=True)

        for i, col in enumerate(self.df.columns[:-1]):
            x = data[i]
            ax = axes[i]
            ax.plot(t_s[:five_min], x[:five_min], label=col)

            y_min = np.min(x[:five_min]) - 10
            y_max = np.max(x[:five_min]) + 10
            ax.set_ylim(y_min, y_max)

            cmap_red = plt.cm.Reds
            groundtruth_labels = self.df['ground_truth'].values
            for j, label in enumerate(groundtruth_labels[:five_min]):
                if label == 1:
                    ax.axvspan(t_s[j], t_s[j + 1], alpha=0.5, color=cmap_red(0.7))

            cmap_blue = plt.cm.Blues
            predicted_labels = self.df['label'].values
            for j, label in enumerate(predicted_labels[:five_min]):
                if label == 1:
                    ax.axvspan(t_s[j], t_s[j + 1], alpha=0.5, color=cmap_blue(0.7))

            ax.legend(loc='upper right')

        plt.xlabel('Time (s)')
        plt.suptitle('EEG Data (First 5 Minutes)')
        plt.tight_layout()
        plt.show()

    def plot_eeg_multiple_optimized(self):
        """
        Optimized plotting of EEG channels as separate subplots with highlighted labels.
        """
        n_samples = len(self.df.index)
        data = self.df.to_numpy().T
        t_s = (self.df.index - self.df.index[0]).total_seconds()
        five_min = next((i for i, t in enumerate(t_s) if t > 300), n_samples)

        fig, axes = plt.subplots(len(self.df.columns[:-1]), 1, figsize=(12, 8), sharex=True)

        for i, col in enumerate(self.df.columns[:-1]):
            x = data[i]
            ax = axes[i]

            ax.plot(t_s[:five_min], x[:five_min], label=col, antialiased=False, linewidth=0.8)

            y_min = np.min(x[:five_min])
            y_max = np.max(x[:five_min])
            y_range = np.ptp(x[:five_min])
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            groundtruth_labels = self.df['ground_truth'].values[:five_min]
            if len(groundtruth_labels) > 0:
                ax.fill_between(t_s[:five_min], y_min, y_max, where=(groundtruth_labels == 1),
                                color='red', alpha=0.5, step='mid')

            predicted_labels = self.df['label'].values[:five_min]
            if len(predicted_labels) > 0:
                ax.fill_between(t_s[:five_min], y_min, y_max, where=(predicted_labels == 1),
                                color='blue', alpha=0.5, step='mid')

            ax.legend(loc='upper right')

        plt.xlabel('Time (s)')
        plt.suptitle('EEG Data (First 5 Minutes)')
        plt.tight_layout()
        plt.show()
