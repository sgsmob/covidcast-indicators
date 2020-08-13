import pytest

from os import listdir
from os.path import join

import pandas as pd
from delphi_nchs_mortality.run import run_module


class TestRun:
    def test_output_files_exist(self, run_as_module):

        csv_files = listdir("receiving")

        dates = [
            "20200411",
            "20200418",
            "20200425",
            "20200502",
            "20200509",
            "20200516",
            "20200523",
            "20200530",
            "20200606",
            "20200613",
            "20200620",
        ]
        metrics = [
                'covid_deaths', 'total_deaths', 'pneumonia_deaths',
                'pneumonia_and_covid_deaths', 'influenza_deaths', 
                'pneumonia_influenza_or_covid_19_deaths'
        ]
        sensors = ["num", "prop"]

        expected_files = []
        for date in dates:
            for metric in metrics:
                for sensor in sensors:
                    expected_files += [date + "_wip_" + metric + "_" + sensor + ".csv"]

        assert set(expected_files).issubset(csv_files)

    def test_output_file_format(self, run_as_module):

        df = pd.read_csv(
            join("receiving", "20200620_wip_covid_deaths_prop.csv")
        )
        assert (df.columns.values == ["geo_id", "val", "se", "sample_size"]).all()