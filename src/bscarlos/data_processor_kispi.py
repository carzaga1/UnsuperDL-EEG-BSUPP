import tqdm
import pandas as pd

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
                         
    def remove_ground_truth_column(self, patient_data):
        for patient_id, patient_info in tqdm.tqdm(patient_data.items()):
            training_data = patient_info['training_data']
            testing_data = patient_info['testing_data']
            
            eeg_training = training_data.drop(columns='ground_truth')
            eeg_testing = testing_data.drop(columns='ground_truth')
            
            patient_data[patient_id]['eeg_training'] = eeg_training
            patient_data[patient_id]['eeg_testing'] = eeg_testing
        return patient_data