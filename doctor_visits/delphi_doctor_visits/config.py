"""
This file contains configuration variables used to generate the doctor visits signal.

Author: Maria
Created: 2020-04-16
Last modified: 2020-05-12
"""

from datetime import datetime, timedelta


class Config:
    """Static configuration variables.
    """

    # dates
    DAY_SHIFT = timedelta(days=1)  # shift dates forward for labeling purposes
    # Feb 1 is when we start producing sensors
    FIRST_SENSOR_DATE = datetime(2020, 2, 1) - DAY_SHIFT
    # add burn-in sensor dates to calculate direction
    BURN_IN_DATE = datetime(2020, 1, 29) - DAY_SHIFT

    # data columns
    CLI_COLS = ["Covid_like", "Flu_like", "Mixed"]
    FLU1_COL = ["Flu1"]
    COUNT_COLS = CLI_COLS + FLU1_COL + ["Denominator"]
    DATE_COL = "ServiceDate"
    GEO_COL = "PatCountyFIPS"
    AGE_COL = "PatAgeGroup"
    HRR_COLS = ["Pat HRR Name", "Pat HRR ID"]
    ID_COLS = [DATE_COL] + [GEO_COL] + [AGE_COL] + HRR_COLS
    FILT_COLS = ID_COLS + COUNT_COLS
    DTYPES = {
        "ServiceDate": str,
        "PatCountyFIPS": str,
        "Denominator": int,
        "Flu1": int,
        "Covid_like": int,
        "Flu_like": int,
        "Mixed": int,
        "PatAgeGroup": str,
        "Pat HRR Name": str,
        "Pat HRR ID": float,
    }

    SMOOTHER_BANDWIDTH = 100  # bandwidth for the linear left Gaussian filter
    MIN_OBS = 2500  # number of total visits needed to produce a sensor
    MAX_BACKFILL_WINDOW = (
        7  # maximum number of days used to average a backfill correction
    )
    MIN_CUM_VISITS = 500  # need to observe at least 500 counts before averaging
    RECENT_LENGTH = 7  # number of days to sum over for sparsity threshold
    MIN_RECENT_VISITS = 100  # min numbers of visits needed to include estimate
    MIN_RECENT_OBS = 3  # minimum days needed to produce an estimate for latest time
    assert MIN_OBS >= MIN_CUM_VISITS, "Backfill adjustment not guaranteed to work"
