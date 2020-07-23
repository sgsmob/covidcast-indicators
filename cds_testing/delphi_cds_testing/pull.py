# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def pull_cds_data(base_url: str, countyname_to_fips_df: pd.DataFrame,
                  pop_df: pd.DataFrame) -> pd.DataFrame:
    """Pulls the latest Testing data from Corona Data Scraper, and conforms
    it into a dataset

    The output dataset has:

    - Each row corresponds to (County, Date), denoted (FIPS, timestamp)
    - Each row additionally has a column `new_counts` corresponding to the new
      new_counts (either `confirmed` cases or `deaths`), and a column
      `cumulative_counts`, correspond to the aggregate metric from January 22nd
      (as of April 27th) until the latest date.

    Note that the raw dataset gives the `cumulative_counts` metric, from which
    we compute `new_counts` by taking first differences.  Hence, `new_counts`
    may be negative.  This is wholly dependent on the quality of the raw
    dataset.

    We replace all of the FIPS in PR to 72000 since we don't have detailed
    population information yet.

    Parameters
    ----------
    base_url: str
        Base URL for pulling the CDS data
    countyname_to_fips.csv: str
        Read from static file which includes the mapping info from county names
        to FIPS code. This is according to the location metadata
        (https://coronadatascraper.com/locations.json) provided by CDS.
    pop_df: pd.DataFrame
        Read from static file "fips_population.csv".

    Returns
    -------
    pd.DataFrame
        Dataframe as described above.
    """
    # Constants
    DROP_COLUMNS = ["population", "level", "city", "county",
                "lat", "long", "url", "name", "country", "state",
                "aggregate", "tz", "cases", "deaths", "recovered",
                "active", "hospitalized", "hospitalized_current",
                "discharged", "icu", "icu_current", "growthFactor"
    ]

    # Read data
    df = pd.read_csv(base_url, low_memory=False, parse_dates=["date"]).rename(
        {"date": "timestamp", "tested": "cumulative_counts"}, axis=1
    )
    # Get fips code
    df = df.merge(countyname_to_fips_df, on="name", how="left")

    df = df[
        (
            (df["country"] == "United States")  # US only
            & (df["level"] == "county") # county level only
        )
    ]

    # Manual correction for PR
    df.loc[df["fips"] // 1000 == 72, "fips"] = 72000

    # Drop unnecessary columns (state is pre-encoded in fips)
    try:
        df.drop(DROP_COLUMNS, axis=1, inplace=True)
    except KeyError as e:
        raise ValueError(
            "Tried to drop non-existent columns. The dataset "
            "schema may have changed.  Please investigate and "
            "amend DROP_COLUMNS."
        )

    # Check that columns are either FIPS or dates
    try:
        columns = list(df.columns)
        columns.remove("fips")
        columns.remove("cumulative_counts")
        columns.remove("timestamp")
        _ = np.vstack((columns, []))
    except ValueError as e:
        print(e)
        raise ValueError(
            "Detected unexpected column(s) "
            "after dropping DROP_COLUMNS. The dataset "
            "schema may have changed. Please investigate and "
            "amend DROP_COLUMNS."
        )

    # Fill in missing dates and combine rows for PR
    FIPS_LIST = df["fips"].unique()
    DATE_LIST = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max())
    index_df = pd.MultiIndex.from_product(
        [FIPS_LIST, DATE_LIST], names=['fips', 'timestamp']
    )
    df = df.groupby(["fips", "timestamp"]).sum().reindex(index_df).reset_index().fillna(0)

    # Merge in population
    df = pd.merge(df, pop_df, on="fips", how='left')

    # Conform FIPS
    df["fips"] = df["fips"].apply(lambda x: f"{int(x):05d}")

    # Add a dummy first row here on day before first day
    min_ts = min(df["timestamp"])
    df_dummy = df.loc[df["timestamp"] == min_ts].copy()
    df_dummy.loc[:, "timestamp"] = min_ts - pd.Timedelta(days=1)
    df_dummy.loc[:, "cumulative_counts"] = 0
    df = pd.concat([df_dummy, df])
    # Obtain new_counts
    df.sort_values(["fips", "timestamp"], inplace=True)
    df["new_counts"] = df["cumulative_counts"].diff()  # 1st discrete difference
    # Handle edge cases where we diffed across fips
    mask = df["fips"] != df["fips"].shift(1)
    df.loc[mask, "new_counts"] = np.nan
    df.reset_index(inplace=True, drop=True)

    # Final sanity checks
    days_by_fips = df.groupby("fips").count()["cumulative_counts"].unique()
    unique_days = df["timestamp"].unique()
    # each FIPS has same number of rows
    if (len(days_by_fips) > 1) or (days_by_fips[0] != len(unique_days)):
        raise ValueError("Differing number of days by fips")
    min_timestamp = min(unique_days)
    max_timestamp = max(unique_days)
    n_days = (max_timestamp - min_timestamp) / np.timedelta64(1, "D") + 1
    if n_days != len(unique_days):
        raise ValueError(
            f"Not every day between {min_timestamp} and "
            "{max_timestamp} is represented."
        )
    return df.loc[
        df["timestamp"] >= min_ts,
        [  # Reorder
            "fips",
            "timestamp",
            "population",
            "new_counts",
            "cumulative_counts",
        ],
    ]