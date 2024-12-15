import tqdm
import pandas as pd
import numpy as np

from scipy.signal import iirfilter, sosfiltfilt
from sklearn.model_selection import KFold

class PatientDataProcessor:
    """
    A class used to process patient data.

    Attributes:
        None

    Methods:
        process_patient_data(patient_data): Processes all patients in the provided patient data dictionary.
        process_patient(patient_data, patient_id, patient_info): Processes a single patient's data.

    Notes:
        The patient data dictionary is expected to have the following structure:
            {
                patient_id: {
                    'data': pandas.DataFrame,
                    'fs': float
                }
            }
        Where 'data' is a pandas DataFrame containing the patient's data, and 'fs' is the sampling frequency.
    """
    def __init__(self):
        pass
    
    def create_patient_data_dict(self, data_files, data_attributes):
        patient_data = {}
        for file in tqdm.tqdm(data_files):
            patient_id = file.stem.split("_")[0] + "_" + file.stem.split("_")[1]
            
            data = pd.read_parquet(file)
            
            fs = data_attributes.loc[patient_id, "sample_rate"]
            
            patient_data[patient_id] = {'data': data, 'fs': fs}
        return patient_data
        
    def apply_timedelta_index(self, patient_data):
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            data = patient_info['data']
            
            fs = patient_info['fs']
            
            Ts = pd.Timedelta('1s') / fs
            
            data.index = data.index * Ts
            
            patient_data[patient_id]['data'] = data
        return patient_data
    
    def apply_bandpass_filter(self, patient_data, low_cut=0.5, high_cut=15.0):
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            data = patient_info['data']
            fs = patient_info['fs']
        
            # Band-pass filter design
            sos = iirfilter(
                 N=18,                  # 18th-order filter
                 Wn=[low_cut, high_cut],  # Frequency range
                 btype='band',          # Band-pass filter
                 ftype='cheby1',        # Chebyshev type I
                 rp=0.1,                # Maximum ripple (0.1 dB)
                 fs=fs,                 # Sampling frequency
                 output='sos'           # Second-order sections
             )
     
             # Identify columns to filter (e.g., exclude 'ground_truth')
            signal_columns = data.select_dtypes(include='number').columns.difference(['ground_truth'])
             
             # Apply the filter only to the signal columns
            filtered_signals = data[signal_columns].apply(lambda x: sosfiltfilt(sos, x), axis=0)
             
             # Combine filtered signals with non-filtered columns
            filtered_data = pd.concat([filtered_signals, data.drop(columns=signal_columns)], axis=1)
             
             # Preserve column order
            filtered_data = filtered_data[data.columns]
             
             # Update patient data with filtered signals
            patient_data[patient_id]['data'] = filtered_data

        return patient_data
    
    def split_data_for_training(self, patient_data):
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            data = patient_info['data']
            
            total_duration = data.index[-1] - data.index[0]
            learning_period = pd.Timedelta(seconds=total_duration.total_seconds() * 0.25)
            
            training_data = data[data.index < learning_period]
            testing_data = data[data.index >= learning_period]
            
            patient_data[patient_id]['training_data'] = training_data
            patient_data[patient_id]['testing_data'] = testing_data
        return patient_data
    
    def split_data_with_15min_learning_period(self, patient_data):
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            data = patient_info['data']
    
            # Calculate the cutoff datetime for learning data (15 minutes)
            learning_period = pd.Timedelta(minutes=15)  
            cutoff_datetime = data.index[0] + learning_period
    
            training_data = data[data.index < cutoff_datetime]
            testing_data = data[data.index >= cutoff_datetime]
    
            patient_data[patient_id]['training_data'] = training_data
            patient_data[patient_id]['testing_data'] = testing_data
        return patient_data
    
    def split_data_for_5fold_cv(self, patient_data, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Initialize KFold with 5 splits
        
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            data = patient_info['data']
            
            # Initialize lists to store train and test splits
            train_folds = []
            test_folds = []
            
            # Iterate over each split
            for train_index, test_index in kf.split(data):
                train_data = data.iloc[train_index]
                test_data = data.iloc[test_index]
                
                train_folds.append(train_data)
                test_folds.append(test_data)
            
            # Store the folds for this patient
            patient_data[patient_id]['train_folds'] = train_folds
            patient_data[patient_id]['test_folds'] = test_folds
        
        return patient_data
                         
    def remove_ground_truth_column(self, patient_data):
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            training_data = patient_info['training_data']
            testing_data = patient_info['testing_data']
            
            eeg_training = training_data.drop(columns='ground_truth')
            eeg_testing = testing_data.drop(columns='ground_truth')
            
            patient_data[patient_id]['eeg_training'] = eeg_training
            patient_data[patient_id]['eeg_testing'] = eeg_testing
        return patient_data