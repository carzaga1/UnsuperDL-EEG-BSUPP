import pandas as pd

from bscarlos.settings import FILENAME_PANTIENT_LIST


def load_patient_list(include_mrns=False):
    return pd.read_csv(FILENAME_PANTIENT_LIST)