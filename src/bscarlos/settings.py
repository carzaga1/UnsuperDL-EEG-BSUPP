"""Module storing settings that are used in this project."""

import os
from pathlib import Path

import yaml


def load_local_settings(filename):
    if not os.path.isfile(filename):
        return None
    
    with open(filename, 'r') as f:
        settings_data = yaml.safe_load(f)
        
    return settings_data


# project folder
PROJECT_FOLDER = Path(__file__).resolve().parents[2]

# load local settings
FILENAME_LOCAL_SETTINGS = PROJECT_FOLDER / 'local_settings.yml'
local_settings = load_local_settings(FILENAME_LOCAL_SETTINGS)

# project sub-folders
if local_settings is not None and 'data_folder' in local_settings:
    DATA_FOLDER = Path(local_settings['data_folder']).resolve()
else:
    DATA_FOLDER = PROJECT_FOLDER / 'data'

RAW_DATA_FOLDER = DATA_FOLDER / 'raw'
PROCESSED_DATA_FOLDER = DATA_FOLDER / 'processed'
RAW_KISPI_DATA_FOLDER = DATA_FOLDER / 'raw_kispi'
PROCESSED_KISPI_DATA_FOLDER = DATA_FOLDER / 'processed_kispi'
REFERENCES_FOLDER = PROJECT_FOLDER / 'references'

# filenames
FILENAME_PANTIENT_LIST = REFERENCES_FOLDER / 'mrn_pseudonym_keys.csv'
FILENAME_DATA_GAGAN = DATA_FOLDER / 'patient_data_2.hdf'