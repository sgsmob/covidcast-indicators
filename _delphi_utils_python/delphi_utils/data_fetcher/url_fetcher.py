"""Definition of UrlDataFetcher."""

import pandas as pd

from .fetcher import DataFetcher

class UrlDataFetcher(DataFetcher):
    def __init__(self,
                 url: str):
        super().__init__()
        self.url = url
        self.df = None

    def setup(self):
        self.df = pd.read_csv(self.url)


    