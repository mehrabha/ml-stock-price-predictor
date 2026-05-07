from datetime import datetime
import numpy as np
import pandas as pd
import os
import requests
import time
import pandas_market_calendars as mkt_cal

class Dataset:
    def __init__(self, ticker: str, start_date: datetime, end_date: datetime) -> None:
        self.ticker = ticker

        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("Dates must be provided as datetime objects")
        
        self.start_dt = start_date
        self.start_date = start_date.strftime("%Y-%m-%d")
        self.end_dt = end_date
        self.end_date = end_date.strftime("%Y-%m-%d")

    def download_csvs(self):
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

            # allow up to 5 calls per minute
            time.sleep(12)
        print('\n')

        return results
    
    def generate_training_dataset(
            self, start_dt: datetime, end_dt: datetime, 
            hourly_lookback_days: int, daily_bars: int, weekly_bars: int, monthly_bars: int, 
            max_news_per_hr: int, label: str):
        
        print(f"----------Generating {self.ticker} dataset for {start_dt}:{end_dt}----------")

        if start_dt < self.start_dt:
            raise ValueError(f"Provided start date{start_dt} preceeds available training data!")
        
        if not (hourly_lookback_days > 1 and daily_bars > hourly_lookback_days and weekly_bars > 1 and monthly_bars > 1):
            raise ValueError(f"Bar count params must be > 1!")
        
        print("Loading CSVs...")

        datasets = ["prices_hour", "prices_day", "prices_week", "prices_month", "news"]
        datasets = {name: self.ticker + '_' + name for name in datasets}

        files = os.listdir("./data/raw")

        # load csvs
        for dataset, name in datasets.items():
            found = False
            for file in files:
                if file.startswith(name):
                    found = True

                    datasets[dataset] = pd.read_csv(f"./data/raw/{file}")
                    print(f"{file} loaded!")
                    break
            if not found:
                raise ValueError(f"./data/raw/{name}.csv not present! Invoke download_csvs() to fetch prices/news")


        print("Sorting csvs", end='', flush=True)
        for name in datasets.keys():
            df = datasets[name].set_index("eastern_dt")
            df.index = pd.to_datetime(df.index, utc=True).tz_convert("US/Eastern")
            df = df.sort_index()
            datasets[name] = df
            print(".", end='', flush=True)
        
        df_hour: pd.DataFrame = datasets["prices_hour"]
        df_day: pd.DataFrame = datasets["prices_day"]
        df_week: pd.DataFrame = datasets["prices_week"]
        df_month: pd.DataFrame = datasets["prices_month"]
        df_news: pd.DataFrame = datasets["news"]

        # trim extra hours
        df_hour = df_hour.between_time("8:00", "18:00", inclusive="left")

        # prep training dataset
        mkt_days = mkt_cal.get_calendar("NYSE").valid_days(start_date=self.start_dt, end_date=end_dt)   # nyse mkt days
        mkt_days = mkt_days.tz_localize(None).tz_localize("US/Eastern").to_list()

        print("Processing rows...",)
        rows = []
        skipped_rows = 0
        start_dt = pd.Timestamp(start_dt).tz_localize("US/Eastern")
        for i in range(len(mkt_days)):
            if i % 10 == 0:
                print('.')

            mkt_dt = mkt_days[i]
            if mkt_dt < start_dt:
                # skip until we are in the training window
                continue

            row = {"date": mkt_dt.strftime("%Y-%m-%d")}

            # get hourly price points for previous n business days
            hourly = []
            bday = i - hourly_lookback_days
            while mkt_days[bday] < mkt_dt:
                start = mkt_days[bday]
                points = df_hour.loc[start.replace(hour=8):start.replace(hour=17, minute=59, second=59)]    # 8am-6pm prices
                if len(points) != 10:
                    print(f"Warning! Found {len(points)} hour price points for {start}, expected 10! This could be due to holidays.")
                    if len(points) == 0 or len(points) > 10:
                        bday += 1
                        continue
                    last_row = points.iloc[[-1]]
                    points = pd.concat([points] + [last_row] * (10 - len(points)), ignore_index=True)
                hourly += points.to_dict("records")
                bday += 1

            # add today's prices
            points = df_hour.loc[mkt_dt.replace(hour=8):mkt_dt.replace(hour=14, minute=59, second=59)]     # 8am-3pm prices
            hourly += points.to_dict("records")

            if len(hourly) != 10 * hourly_lookback_days + 7:
                print(f"Warning! Found {len(hourly)} hour price points for window starting:{mkt_dt.date()}, expected {10 * hourly_lookback_days + 7}! Skipping row...")
                skipped_rows += 1
                continue
            row["hour"] = hourly

            # append daily bars for previous n business days
            mkt_dt_start = mkt_days[i - daily_bars]
            mkt_dt_end = mkt_days[i - 1]
            daily = df_day.loc[mkt_dt_start:mkt_dt_end]
            if len(daily) != daily_bars:
                print(f"Warning! Found {len(daily)} prev day bars for {mkt_dt}, expected {daily_bars}! Check datasets..")
                skipped_rows += 1
                continue
            row["day"] = daily.to_dict("records")

            # append n weekly bars
            end = mkt_dt - pd.offsets.Week(n=1, weekday=5)    # last saturday
            start = end - pd.offsets.Week(n=weekly_bars, weekday=6)     # starting sunday from n weeks ago
            weekly = df_week.loc[start:end]
            if len(weekly) != weekly_bars:
                print(f"Warning! Found {len(weekly)} prev weekly bars for {mkt_dt}, expected {weekly_bars}! Check datasets..")
                skipped_rows += 1
                continue
            row["week"] = weekly.to_dict("records")

            # append the monthly bars
            end = mkt_dt.replace(day=1) - pd.Timedelta(days=1)
            start = mkt_dt.replace(day=1) - pd.offsets.MonthBegin(monthly_bars)
            monthly = df_month.loc[start:end]
            if len(monthly) != monthly_bars:
                print(f"Warning, Found {len(monthly)} prev monthly bars for {mkt_dt}, expected {monthly_bars}! Check datasets..")
                skipped_rows += 1
                continue
            row["month"] = monthly.to_dict("records")


            # append hourly news
            start = mkt_days[i - 1].replace(hour=15)    # starting 3pm prev business day
            end = mkt_dt.replace(hour=15)   # 3pm today

            news = []
            news_found = False
            while start < end:  # 3pm-7pm prev day, 7am-2:59pm
                news_hour = df_news.loc[start: start.replace(hour=start.hour+1, minute=59, second=59)]     # 2 hr interval
                news_hour_count = len(news_hour)

                if news_hour_count > 0:
                    news_found = True

                # keep up to n news in case there are more
                if len(news_hour) >= max_news_per_hr * 2:
                    news_hour = news_hour.sample(n=max_news_per_hr * 2)

                news.append({"count": news_hour_count, "samples": news_hour.to_dict("records")})    # save as (total count, sampled)
                
                start = start.replace(hour=start.hour + 2)
                if start.hour >= 19:
                    # go to next business day 7am
                    start = end.replace(hour=7)

            if not news_found:
                print(f"Warning, No news available in the last 24 hrs for {mkt_dt.date()}!")

            # add prev weekend news if available for extra context
            saturday_dt = mkt_dt - pd.offsets.Week(n=1, weekday=5)
            saturday_news = df_news.loc[saturday_dt.strftime("%Y-%m-%d")]
            saturday_count = len(saturday_news)

            sunday_dt = mkt_dt - pd.offsets.Week(n=1, weekday=6)
            sunday_news = df_news.loc[sunday_dt.strftime("%Y-%m-%d")]
            sunday_count = len(sunday_news)

            if len(saturday_news) > max_news_per_hr * 2:
                saturday_news = saturday_news.sample(n=max_news_per_hr * 2)     # using hourly rate for weekends to keep things proportional
            if len(sunday_news) > max_news_per_hr * 2:
                sunday_news = sunday_news.sample(n=max_news_per_hr * 2)     # weird rate but keeps things proportional
            news.append({"count": saturday_count, "samples": saturday_news.to_dict("records")})
            news.append({"count": sunday_count, "samples": sunday_news.to_dict("records")})

            if len(news) != 8:
                print(f"Exception, expected 8 objects in news array, found {len(news)}! {mkt_dt}...")
                skipped_rows += 1
                continue
            else:
                row["news"] = news

            # the main dish
            if i != len(mkt_days) - 1:
                next_dt = mkt_days[i + 1]

                day_bar = df_hour.loc[mkt_dt.replace(hour=14):mkt_dt.replace(hour=14, minute=59, second=59)]
                next_day_bar = df_hour.loc[next_dt.replace(hour=14):next_dt.replace(hour=14, minute=59, second=59)]

                if len(day_bar) != 1:
                    print(f"Warning, day bar has {len(day_bar)} points, expected 1. Proceeding...")
                    skipped_rows += 1
                    continue
                if len(next_day_bar) != 1:
                    print(f"Warning, next day bar has {len(next_day_bar)} points, expected 1. Proceeding...")
                    skipped_rows += 1
                    continue

                closing_price = day_bar.iloc[0]["close"]
                next_day_price = next_day_bar.iloc[0]["close"]

                row["target"] = 1 if next_day_price > closing_price else 0
            else:
                # guess 1 to handle edge case, should be a one time thing
                row["target"] = 1
        
            if len(row) != 7:
                print(f"Exception, expected 7 elements in row, found {len(row)}! Aborting...")
                return
            rows.append(row)
        
        print(f"Extracted {len(rows)} rows! Skipped {skipped_rows} rows")

        # save as parquet
        os.makedirs("./data/processed", exist_ok=True)
        filepath = f"./data/processed/{label}.parquet"

        if os.path.exists(filepath):
            print(f"Parquet {filepath} found! Replacing...")

        final_df = pd.DataFrame(rows)
        final_df.to_parquet(filepath, index=False, engine="pyarrow")

        print(f"Success: {filepath}!")


                




