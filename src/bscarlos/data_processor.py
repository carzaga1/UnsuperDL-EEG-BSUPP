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

    def process_patient_data(self, patient_data):
        for patient_id, patient_info in tqdm.tqdm(self.patient_data.items()):
            self.process_patient(patient_data, patient_id, patient_info)

    def process_patient(self, patient_data, patient_id, patient_info):
        data = patient_info['data']
        fs = patient_info['fs']

        Ts = pd.Timedelta('1s') / fs
        data.index = data.index * Ts
        self.patient_data[patient_id]['data'] = data

        learning_period = pd.Timedelta(seconds=60 * 15)
        data_learning = data[data.index < learning_period]
        data_testing = data[data.index >= learning_period]

        patient_data[patient_id]['data_learning'] = data_learning
        patient_data[patient_id]['data_testing'] = data_testing