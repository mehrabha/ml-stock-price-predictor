import os
from datetime import datetime
import pandas as pd
import pandas_market_calendars as mkt_cal

class LLMStockTrader:
    def __init__(self, starting_balance: int, max_trades_per_day=3):
        # Portfolio
        self.cash_balance = starting_balance
        self.holdings = {}
        self.trade_history = []

        self.max_trades_per_day = max_trades_per_day
        self.daily_trades_executed = 0

        self.llm_url = f"http://localhost:5000{os.getenv("BASE_PATH")}/chat"
        self.llm_auth = (os.getenv("APP_USER"), os.getenv("APP_PASS"))

    def observe(
            self, ticker: str,
            hourly_lookback_days: int, daily_bars: int, 
            weekly_bars: int, monthly_bars: int, news_articles: int):
        """
        Step 1: Fetch live prices and news
        """

        now = pd.Timestamp.now()
        print(f"[OBSERVE] Fetching latest data for {timestamp}...")


        # get valid business days
        mkt_days = mkt_cal.get_calendar("NYSE").valid_days(start_date=now - pd.DateOffset(months=6), end_date=now)   # nyse mkt days
        mkt_days = mkt_days.tz_localize(None).tz_localize("US/Eastern").to_list()