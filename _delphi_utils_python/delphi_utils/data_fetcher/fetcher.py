"""Interface for data fetching."""

import datetime

class DataFetcher:
    """Abstract base class for data fetching."""
    def __init__(self):
        pass

    def setup(self):
        """Set up connection."""
        raise NotImplementedError

    def fetch(self, start_date: datetime.date, end_date: datetime.date):
        """
        Fetch data from the source.
        Params
        ------
        start_date: datetime.date
            Earliest date from which to fetch data
        end_date: datetime.date
            Latest date from which to fetch data
        """
        raise NotImplementedError
