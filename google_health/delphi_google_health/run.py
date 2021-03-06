# -*- coding: utf-8 -*-
"""Functions to call when running the function.

This module should contain a function called `run_module` that is executed
when the module is run with `python -m MODULE_NAME`.
"""

import datetime
import logging
import time

import pandas as pd

from delphi_utils import (
    S3ArchiveDiffer,
    add_prefix,
    create_export_csv,
    get_structured_logger
)

from .data_tools import format_for_export
from .pull_api import GoogleHealthTrends, get_counts_states, get_counts_dma
from .map_values import derived_counts_from_dma
from .constants import (SIGNALS, SMOOTHED,
                        MSA, HRR, STATE, DMA,
                        PULL_START_DATE)


def run_module(params):
    """
    Main function run when calling the module.

    Inputs parameters from the file 'params.json' and produces output data in
    the directory defined by the `export_dir` (should be "receiving" expect for
    testing purposes).

    Parameters
    ----------
    params
        Dictionary containing indicator configuration. Expected to have the following structure:
        - "common":
            - "export_dir": str, directory to write output
            - "log_exceptions" (optional): bool, whether to log exceptions to file
            - "log_filename" (optional): str, name of file to write logs
        - "indicator":
            - "static_file_dir": str, path to DMA mapping files
            - "data_dir": str, location of cached CSVs
            - "start_date": str, YYYY-MM-DD format, first day to generate data for
            - "end_date": str, YYYY-MM-DD format or empty string, last day to generate data for.
            - "ght_key": str, GHT API key
            - "wip_signal": list of str or bool, to be passed to delphi_utils.add_prefix
            - "test_data_dir": str, path to test data
        - "archive" (optional): if provided, output will be archived with S3
            - "aws_credentials": Dict[str, str], AWS login credentials (see S3 documentation)
            - "bucket_name: str, name of S3 bucket to read/write
            - "cache_dir": str, directory of locally cached data
    """
    start_time = time.time()
    csv_export_count = 0
    oldest_final_export_date = None

    # read parameters
    ght_key = params["indicator"]["ght_key"]
    start_date = params["indicator"]["start_date"]
    end_date = params["indicator"]["end_date"]
    static_dir = params["indicator"]["static_file_dir"]
    export_dir = params["common"]["export_dir"]
    data_dir = params["indicator"]["data_dir"]
    wip_signal = params["indicator"]["wip_signal"]

    logger = get_structured_logger(
        __name__, filename=params["common"].get("log_filename"),
        log_exceptions=params["common"].get("log_exceptions", True))

    if "archive" in params:
        arch_diff = S3ArchiveDiffer(
            params["archive"]["cache_dir"], export_dir,
            params["archive"]["bucket_name"], "ght",
            params["archive"]["aws_credentials"])
        arch_diff.update_cache()

    # if missing start_date, set to today (GMT) minus 5 days
    if start_date == "":
        now = datetime.datetime.now(datetime.timezone.utc)
        start_date = (now - datetime.timedelta(days=4)).strftime("%Y-%m-%d")

    # if missing end_date, set to today (GMT) minus 5 days
    if end_date == "":
        now = datetime.datetime.now(datetime.timezone.utc)
        end_date = (now - datetime.timedelta(days=4)).strftime("%Y-%m-%d")

    # Turn on basic logging messages (level INFO)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    logging.info("Creating data from %s through %s.", start_date, end_date)

    # Dictionary mapping geo resolution to the data corresponding to that resolution.
    df_by_geo_res = {}

    if not params["indicator"]["test_data_dir"]:
        # setup class to handle API calls
        ght = GoogleHealthTrends(ght_key=ght_key)

        # read data frame version of the data
        df_by_geo_res[STATE] = get_counts_states(
            ght, PULL_START_DATE, end_date, static_dir=static_dir, data_dir=data_dir
        )
        df_by_geo_res[DMA] = get_counts_dma(
            ght, PULL_START_DATE, end_date, static_dir=static_dir, data_dir=data_dir
        )
    else:
        df_by_geo_res[STATE] = pd.read_csv(
            params["indicator"]["test_data_dir"].format(geo_res="state"))
        df_by_geo_res[DMA] = pd.read_csv(params["indicator"]["test_data_dir"].format(geo_res="dma"))

    df_by_geo_res[HRR], df_by_geo_res[MSA] = derived_counts_from_dma(df_by_geo_res[DMA],
                                                                     static_dir=static_dir)

    signal_names = add_prefix(SIGNALS, wip_signal, prefix="wip_")

    for signal in signal_names:
        is_smoothed = signal.endswith(SMOOTHED)
        for geo_res, df in df_by_geo_res.items():
            exported_csv_dates = create_export_csv(
                                    format_for_export(df, is_smoothed),
                                    geo_res=geo_res,
                                    sensor=signal,
                                    start_date=start_date,
                                    export_dir=export_dir)

            if not exported_csv_dates.empty:
                csv_export_count += exported_csv_dates.size
                if not oldest_final_export_date:
                    oldest_final_export_date = max(exported_csv_dates)
                oldest_final_export_date = min(
                    oldest_final_export_date, max(exported_csv_dates))

    if "archive" in params:
        archive(arch_diff)

    elapsed_time_in_seconds = round(time.time() - start_time, 2)
    max_lag_in_days = None
    formatted_oldest_final_export_date = None
    if oldest_final_export_date:
        max_lag_in_days = (datetime.datetime.now() - oldest_final_export_date).days
        formatted_oldest_final_export_date = oldest_final_export_date.strftime("%Y-%m-%d")
    logger.info("Completed indicator run",
        elapsed_time_in_seconds = elapsed_time_in_seconds,
        csv_export_count = csv_export_count,
        max_lag_in_days = max_lag_in_days,
        oldest_final_export_date = formatted_oldest_final_export_date)


def archive(arch_diff):
    """
    Perform archiving of new results.

    Parameters
    ----------
    arch_diff: ArchiveDiffer
        archiver with an updated cache
    """
    # Diff exports, and make incremental versions
    _, common_diffs, new_files = arch_diff.diff_exports()

    # Archive changed and new files only
    to_archive = [f for f, diff in common_diffs.items() if diff is not None]
    to_archive += new_files
    _, fails = arch_diff.archive_exports(to_archive)

    # Filter existing exports to exclude those that failed to archive
    succ_common_diffs = {f: diff for f, diff in common_diffs.items() if f not in fails}
    arch_diff.filter_exports(succ_common_diffs)

    # Report failures: someone should probably look at them
    for exported_file in fails:
        print(f"Failed to archive '{exported_file}'")
