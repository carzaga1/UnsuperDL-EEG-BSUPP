import numpy as np
import pandas as pd
import tqdm

class DataPreprocessing:
    def __init__(self):
        pass
    
    def create_patient_data_dict(self, data_files, data_attributes):
        patient_data = {}
        for file in tqdm.tqdm(data_files):
            patient_id = file.stem.split("-")[0]
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
                         
    def split_data_for_training(self, patient_data):
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            data = patient_info['data']
            learning_period = pd.Timedelta(seconds=60 * 15)
            training_data = data[data.index < learning_period]
            testing_data = data[data.index >= learning_period]
            patient_data[patient_id]['training_data'] = training_data
            patient_data[patient_id]['testing_data'] = testing_data
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
