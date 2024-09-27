import logging
from logging.handlers import RotatingFileHandler

import click

import bscarlos.data.download_raw_data as download_raw_data
import bscarlos.data.preprocess_data as preprocess_data


@click.group()
@click.option('--debug', is_flag=True)
@click.option('--log-file', 'log_file', type=str, help='If specifed, logging to given file is enabled.')
def main(debug, log_file):
    """This is the main starting point to reproduce the results of this study/project.
    
    \b
    Call the subcommands of the CLI in the following order:
    \b
    1. download-raw-data
    2. preprocess-data
    """

    # configure logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = []
    # command line logging
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)
    
    if debug:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    
    # Logging to file (useful when command is started as service)
    if log_file:
        file_handler = RotatingFileHandler(log_file, backupCount=6, maxBytes=1000000)
        file_handler.setLevel(logging.INFO)
        handlers.append(file_handler)
    
    logging.basicConfig(level=logging.DEBUG, format=log_fmt, handlers=handlers)
    

main.add_command(download_raw_data.main, 'download-raw-data')
main.add_command(preprocess_data.main, 'preprocess-data')


if __name__ == '__main__':
    main()
