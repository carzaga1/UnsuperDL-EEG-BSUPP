import click
import logging


DEBUG = True  # only relevant if called as __main__


@click.command()
def main():
    """Preprocessing data.
    """
    
    logger = logging.getLogger(__name__)
   
    logger.info("Start processing data...")

    logger.info("finished!")
       

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if DEBUG:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
        
    logging.basicConfig(level=logging_level, format=log_fmt)

    main()
