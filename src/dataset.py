from datetime import datetime
import numpy as np
import pandas as pd
import os
import requests
import time

class Dataset:
    def __init__(self, ticker: str, start_date: datetime, end_date: datetime) -> None:
        self.ticker = ticker

        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("Dates must be provided as datetime objects")
        
        self.start_date = start_date.strftime("%Y-%m-%d")
        self.end_date = end_date.strftime("%Y-%m-%d")

    def get_csvs(self):
        timespans = ["hour", "day", "week", "month"]

        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not set! Check .env")

        for timespan in timespans:
            print(f"----------Downloading {self.ticker} prices by {timespan}----------")

            filename = f"{self.ticker}_prices_{timespan}_{self.start_date}_{self.end_date}.csv"
            filepath = os.path.join("data/raw", filename)

            # Download csv if not present
            if os.path.isfile(filepath):
                print(f"'data/raw/{filename}' already present, skipping!\n")
            else:       
                self.get_prices(timespan)

        print(f"----------Downloading {self.ticker} news----------")
        filename = f"{self.ticker}_news_{self.start_date}_{self.end_date}.csv"
        filepath = os.path.join("data/raw", filename)

        if os.path.isfile(filepath):
            print(f"'data/raw/{filename}' already present, skipping!\n")
        else:
            self.get_news()


    def get_prices(self, timespan: str):
        api_key = os.getenv("API_KEY")

        if timespan not in ["hour", "day", "week", "month"]:
            raise ValueError("Allowed values for timespan: hour, day, week, month")
        
        start_date = self.start_date
        end_date = self.end_date
        
        os.makedirs("data/raw", exist_ok=True)

        # prepare query to invoke prices api
        url = f"https://api.massive.com/v2/aggs/ticker/{self.ticker}/range/1/{timespan}/{start_date}/{end_date}"

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key
        }

        print(f"Downloading prices for ticker={self.ticker} from {start_date} to {end_date}...", end='', flush=True)

        # download via pagination
        results = self.invoke_apis(url, params, api_key)

        if not results:
            print(f"No results for URL:{url}")
            return
        else:
            print(f"Download completed! {len(results)} rows")
        

        df = pd.DataFrame(results)

        # convert unix timestamps to US Eastern
        df["eastern_dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df["eastern_dt"] = df["eastern_dt"].dt.tz_convert("US/Eastern")

        # rename columns for better readability
        df = df.rename(columns={
            "eastern_dt": "eastern_dt",
            'o': "open",
            'h': "high",
            'l': "low",
            'c': "close",
            'v': "volume",
            'vw': "vwap",
            'n': "transactions"
        })

        # reorder for readability
        df = df[[
            "eastern_dt",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "transactions"
        ]]

        # save
        df.to_csv(f"data/raw/{self.ticker}_prices_{timespan}_{start_date}_{end_date}.csv", index=False)
        print(f"Success: data/raw/{self.ticker}_prices_{timespan}_{start_date}_{end_date}.csv!\n")
    

    def get_news(self):
        api_key = os.getenv("API_KEY")

        url = f"https://api.massive.com/v2/reference/news"

        start_date = self.start_date
        end_date = self.end_date

        # prep query 
        params = {
            "ticker": self.ticker,
            "order": "asc",
            "sort": "published_utc",
            "published_utc.gte": start_date,
            "published_utc.lte": end_date,
            "limit": 1000,
            "apiKey": api_key
        }


        print(f"Invoking news API for {start_date} to {end_date}...", end='', flush=True)

        # download via pagination
        results = self.invoke_apis(url, params, api_key)

        if not results:
            print(f"No results for URL:{url}")
            return
        else:
            print(f"Download completed! {len(results)} rows")
        

        # parse EST times from UTC
        df = pd.DataFrame(results)

        df["eastern_dt"] = pd.to_datetime(df["published_utc"], utc=True)
        df["eastern_dt"] = df["eastern_dt"].dt.tz_convert("US/Eastern")

        # cleanup
        df = df.drop(columns=["amp_url", "image_url", "article_url"], errors="ignore")

        df["publisher"] = df["publisher"].str.get("name")

        # reorder cols for readability
        df = df[[
            "eastern_dt",
            "title",
            "publisher",
            "author",
            "description",
            "keywords",
            "published_utc",
            "tickers",
            "insights"
        ]]

        df.to_csv(f"data/raw/{self.ticker}_news_{start_date}_{end_date}.csv", index=False)
        print(f"Success: data/raw/{self.ticker}_news_{start_date}_{end_date}.csv!\n")

    def invoke_apis(self, base_url, params, api_key):
        results = []
        current_url = base_url

        # download with pagination
        while current_url:
            response = requests.get(current_url, params=params if current_url == base_url else {"apiKey": api_key})
            response.raise_for_status()

            response_json = response.json()

            if "results" in response_json and response_json["results"]:
                results += response_json["results"]
            else:
                print("\nWarning, response has no datapoints")

            print('.', end='', flush=True)
            current_url = response_json.get("next_url")

            # allow up to 5 calls per minute
            time.sleep(12)
        print('\n')

        return results

