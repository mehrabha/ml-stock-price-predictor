import os
from datetime import datetime
import pandas as pd
import pandas_market_calendars as mkt_cal
import src.api_client as polygon_api

class LLMStockTrader:
    def __init__(self, starting_balance: int, max_trades_per_day=3):
        # Portfolio
        self.cash_balance = starting_balance
        self.holdings = {}
        self.trade_history = []
        self.action_summaries = []

        self.max_trades_per_day = max_trades_per_day
        self.daily_trades_executed = 0

        self.llm_url = f"http://localhost:5000{os.getenv('BASE_PATH')}/chat"
        self.llm_auth = (os.getenv("APP_USER"), os.getenv('APP_PASS'))


    def observe(
            self, ticker: str,
            hourly_lookback_days: int, daily_bars: int, 
            weekly_bars: int, monthly_bars: int, news_articles: int,
            trade_history: int, trade_summaries: int):
        """
        Step 1: Fetch live prices and news
        """

        now = pd.Timestamp.now(tz="US/Eastern")
        end_date = now.strftime("%Y-%m-%d")
        print(f"[OBSERVE] Fetching latest prices and market events for {ticker}...")

        # get valid nyse market days
        mkt_days = mkt_cal.get_calendar("NYSE").valid_days(start_date=now - pd.DateOffset(months=6), end_date=now)
        mkt_days = mkt_days.tz_localize(None).tz_localize("US/Eastern").to_list()

        if now.normalize() not in mkt_days:
            raise ValueError("Stock Market is closed today!")
    
        # go back X trading days
        hourly_start = mkt_days[-(hourly_lookback_days + 1)].strftime("%Y-%m-%d")
        daily_start = mkt_days[-(daily_bars + 2)].strftime("%Y-%m-%d")
        weekly_start = (now - pd.DateOffset(weeks=weekly_bars + 2)).strftime("%Y-%m-%d")
        monthly_start = (now - pd.DateOffset(months=monthly_bars + 2)).strftime("%Y-%m-%d")

        # Fetch live data
        try:
            df_hour = polygon_api.get_prices(ticker, "hour", hourly_start, end_date)
            df_day = polygon_api.get_prices(ticker, "day", daily_start, end_date)
            df_week = polygon_api.get_prices(ticker, "week", weekly_start, end_date)
            df_month = polygon_api.get_prices(ticker, "month", monthly_start, end_date)
            df_news = polygon_api.get_news(ticker, hourly_start, end_date)
        except Exception as e:
            raise Exception(f"Error invoking price APIs; cause:{e}")

        # trim extra hours
        if df_hour is not None and not df_hour.empty:
            df_hour = df_hour.set_index("eastern_dt")
            df_hour = df_hour.between_time("8:00", "18:00", inclusive="left")
            df_hour.reset_index()

        if df_news is not None and not df_news.empty:
            df_news = df_news[["eastern_dt", "title", "publisher", "description"]].sort_values(by="eastern_dt")
            md_news = df_news.tail(news_articles).to_markdown(index=False)
        else:
            md_news = f"No news found for window {hourly_start}:{now}!"

        # filter out unnecessary data and save as markdown
        cols = ["eastern_dt", "close", "volume"]
        md_hr = df_hour[cols].tail(hourly_lookback_days * 10)
        md_day = df_day[cols].tail(daily_bars)
        md_week = df_week[cols].tail(weekly_bars)
        md_month = df_month[cols].tail(monthly_bars)


        # check if we were able to fetch the required data
        status = (
            len(md_hr) == hourly_lookback_days * 10 and
            len(md_day) == daily_bars and
            len(md_week) == weekly_bars and
            len(md_month) == monthly_bars
        )

        if status:
            md_hr = md_hr.to_markdown(index=False)
            md_day = md_day.to_markdown(index=False)
            md_week = md_week.to_markdown(index=False)
            md_month = md_month.to_markdown(index=False)
        else:
            raise ValueError(f"Insufficient data for {ticker}; API returned fewer price points...")

        # latest price
        current_price = df_hour["close"].iloc[-1]

        # construct the final context payload
        context = f"""
        **Current Market State for {ticker}**
        - Current Portfolio Cash: ${self.cash_balance:.2f}
        - Current Holdings (Shares): {self.holdings}
        - Trade History: {self.trade_history[-trade_history:] if len(self.trade_history) >= trade_history else self.trade_history}
        - Previous Action Summaries: {self.action_summaries[-trade_summaries:] if len(self.action_summaries) >= self.action_summaries else self.action_summaries}
        - Clock: {now}

        ### Recent News
        {md_news}

        ### Hourly Price Action
        {md_hr}

        ### Daily Trend
        {md_day}

        ### Weekly Structure
        {md_week}

        ### Macro View
        {md_month}
        """
        
        return context, current_price
