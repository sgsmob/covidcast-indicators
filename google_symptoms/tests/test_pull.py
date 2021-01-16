import pytest
import mock
from freezegun import freeze_time
from datetime import date, datetime
import pandas as pd

from delphi_google_symptoms.pull import pull_gs_data, preprocess, get_missing_dates, format_dates_for_query
from delphi_google_symptoms.constants import METRICS, COMBINED_METRIC

base_url_good = "./test_data{sub_url}small_{state}symptoms_dataset.csv"

base_url_bad = {
    "missing_cols": "test_data/bad_state_missing_cols.csv",
    "invalid_fips": "test_data/bad_county_invalid_fips.csv"
}


class TestPullGoogleSymptoms:
    def test_good_file(self):
        dfs = pull_gs_data(base_url_good)

        for level in set(["county", "state"]):
            df = dfs[level]
            assert (
                df.columns.values
                == ["geo_id", "timestamp"] + METRICS + [COMBINED_METRIC]
            ).all()

            # combined_symptoms is nan when both Anosmia and Ageusia are nan
            assert sum(~df.loc[
                (df[METRICS[0]].isnull())
                & (df[METRICS[1]].isnull()), COMBINED_METRIC].isnull()) == 0
            # combined_symptoms is not nan when either Anosmia or Ageusia isn't nan
            assert sum(df.loc[
                (~df[METRICS[0]].isnull())
                & (df[METRICS[1]].isnull()), COMBINED_METRIC].isnull()) == 0
            assert sum(df.loc[
                (df[METRICS[0]].isnull())
                & (~df[METRICS[1]].isnull()), COMBINED_METRIC].isnull()) == 0

    def test_missing_cols(self):
        df = pd.read_csv(base_url_bad["missing_cols"])
        with pytest.raises(KeyError):
            preprocess(df, "state")

    def test_invalid_fips(self):
        df = pd.read_csv(base_url_bad["invalid_fips"])
        with pytest.raises(AssertionError):
            preprocess(df, "county")


class TestPullHelperFuncs:
    @freeze_time("2021-01-05")
    def test_get_missing_dates(self):
        output = get_missing_dates(
            "./test_data/static_receiving", date(year=2020, month=12, day=30))

        expected = [date(2020, 12, 30), date(2021, 1, 4), date(2021, 1, 5)]
        assert set(output) == set(expected)

    def test_format_dates_for_query(self):
        date_list = [date(2016, 12, 30), date(2020, 12, 30),
                     date(2021, 1, 4), date(2021, 1, 5)]
        output = format_dates_for_query(date_list)

        expected = {2020: 'timestamp("2020-12-30")',
                    2021: 'timestamp("2021-01-04"), timestamp("2021-01-05")'}
        assert output == expected
