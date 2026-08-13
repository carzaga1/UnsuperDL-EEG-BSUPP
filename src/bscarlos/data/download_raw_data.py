import click
import logging
import os
import pandas as pd

from bscarlos.helper import load_patient_list

from bscarlos.settings import RAW_DATA_FOLDER

DEBUG = True  # only relevant if called as __main__


@click.command()
def main():
    """Download data.
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Loading patient list.")
    patient_list = load_patient_list()
    logger.info("Start downloading data...")

    for pseudonym in patient_list['pseudonym']:
        logger.info(f"\tLoading data for { pseudonym }.")
        filename_out = RAW_DATA_FOLDER / f"{ pseudonym }-data.parquet"

        if os.path.isfile(filename_out):
            continue
        
        # store data
        df = pd.DataFrame(data={'value': [0,2,4], 'timestamp': pd.date_range(pd.Timestamp.now(), periods=3, freq='1s')})
        df.to_parquet(filename_out)

    logger.info("finished!")
       

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if DEBUG:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
        
    logging.basicConfig(level=logging_level, format=log_fmt)

    main()
