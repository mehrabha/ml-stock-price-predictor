import time
import requests
import os
import pandas as pd

def get_prices(ticker: str, timespan: str, start_date: str, end_date: str) -> pd.DataFrame:
    api_key = os.getenv("API_KEY")

    if timespan not in ["hour", "day", "week", "month"]:
        raise ValueError("Allowed values for timespan: hour, day, week, month")
    
    os.makedirs("data/raw", exist_ok=True)

    # prepare query to invoke prices api
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }

    print(f"Downloading prices for ticker={ticker} from {start_date} to {end_date}...", end='', flush=True)

    # download via pagination
    results = invoke_apis(url, params, api_key)

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

    return df


def get_news(ticker: str, start_date, end_date):
    api_key = os.getenv("API_KEY")

    url = f"https://api.massive.com/v2/reference/news"

    # prep query 
    params = {
        "ticker": ticker,
        "order": "asc",
        "sort": "published_utc",
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "limit": 1000,
        "apiKey": api_key
    }

    print(f"Invoking news API for {start_date} to {end_date}...", end='', flush=True)

    # download via pagination
    results = invoke_apis(url, params, api_key)

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

    return df


def invoke_apis(base_url, params, api_key) -> list:
    results = []
    current_url = base_url

    # download with pagination
    while current_url:
        response = requests.get(
            current_url, 
            params=params if current_url == base_url else {"apiKey": api_key}, 
            timeout=15
        )
        response.raise_for_status()

        response_json = response.json()

        if "results" in response_json and response_json["results"]:
            results += response_json["results"]
        else:
            print("\nWarning, response has no datapoints")

        print('.', end='', flush=True)
        current_url = response_json.get("next_url")

        # allow up to x calls per minute
        time.sleep(1)
    print('\n')

    return results