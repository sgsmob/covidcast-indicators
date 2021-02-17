# -*- coding: utf-8 -*-

from boto3 import Session
from freezegun import freeze_time
from moto import mock_s3
import pytest

from os import listdir, remove
from os.path import join
from shutil import copy

from delphi_nchs_mortality.run import run_module


PARAMS = {
  "common": {
    "export_dir": "./receiving",
    "daily_export_dir": "./daily_receiving"
  },
  "indicator": {
    "daily_cache_dir": "./daily_cache",
    "export_start_date": "2020-04-11",
    "mode":"test_data.csv",
    "static_file_dir": "../static",
    "token": ""
  },
  "archive": {
    "aws_credentials": {
      "aws_access_key_id": "FAKE_TEST_ACCESS_KEY_ID",
      "aws_secret_access_key": "FAKE_TEST_SECRET_ACCESS_KEY"
    },
    "bucket_name": "test-bucket",
    "cache_dir": "./cache"
  }
}

@pytest.fixture(scope="function")
def run_as_module(date):
    # Clean directories
    for fname in listdir("receiving"):
        if ".csv" in fname:
            remove(join("receiving", fname))

    for fname in listdir("cache"):
        if ".csv" in fname:
            remove(join("cache", fname))

    for fname in listdir("daily_cache"):
        if ".csv" in fname:
            remove(join("daily_cache", fname))

    # Simulate the cache already being partially populated
    copy("test_data/weekly_202025_state_wip_deaths_covid_incidence_prop.csv", "daily_cache")

    for fname in listdir("daily_receiving"):
        if ".csv" in fname:
            remove(join("daily_receiving", fname))

    with mock_s3():
        with freeze_time(date):
            # Create the fake bucket we will be using
            aws_credentials = PARAMS["archive"]["aws_credentials"]
            s3_client = Session(**aws_credentials).client("s3")
            s3_client.create_bucket(Bucket=PARAMS["archive"]["bucket_name"])
            run_module(PARAMS)
