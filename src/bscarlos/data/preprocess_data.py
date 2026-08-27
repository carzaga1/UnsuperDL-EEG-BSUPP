import logging

import click

from bscarlos.data.bipolar import create_bipolar_channels
from bscarlos.data.edf_ingest import drop_nonrelevant_channels, find_annotation_edf_files, read_edf_raw
from bscarlos.data.ground_truth import build_ground_truth_labels, parse_annotations
from bscarlos.settings import PROCESSED_KISPI_DATA_FOLDER, RAW_KISPI_DATA_FOLDER

DEBUG = True  # only relevant if called as __main__

ANNOTATORS = ["a1", "a2"]


def _process_one_recording(edf_path, annotator, logger):
    patient_code = edf_path.stem.split("_annotated")[0]

    annotations_path = edf_path.with_name(f"{edf_path.stem}_annotations.txt")
    if not annotations_path.exists():
        logger.warning(f"No annotations file found for {edf_path}, skipping.")
        return

    raw = read_edf_raw(edf_path)
    raw = drop_nonrelevant_channels(raw)

    df = raw.to_data_frame(time_format="datetime", index="time")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    annotations = parse_annotations(annotations_path)
    ground_truth = build_ground_truth_labels(df.index, annotations)

    bipolar_df = create_bipolar_channels(df)
    bipolar_df["ground_truth"] = ground_truth.to_numpy()
    bipolar_df.reset_index(drop=True, inplace=True)

    output_path = PROCESSED_KISPI_DATA_FOLDER / f"{patient_code}_{annotator}_filt_merged.parquet"
    PROCESSED_KISPI_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    bipolar_df.to_parquet(output_path)
    logger.info(f"Wrote {output_path}")


@click.command()
def main():
    """Preprocessing data.
    """

    logger = logging.getLogger(__name__)

    logger.info("Start processing data...")

    for annotator in ANNOTATORS:
        edf_files = find_annotation_edf_files(RAW_KISPI_DATA_FOLDER, annotator)
        for edf_path in edf_files:
            _process_one_recording(edf_path, annotator, logger)

    logger.info("finished!")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if DEBUG:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    logging.basicConfig(level=logging_level, format=log_fmt)

    main()
