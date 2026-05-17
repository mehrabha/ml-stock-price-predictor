import os
from datetime import datetime
import pandas as pd
import pandas_market_calendars as mkt_cal
import src.api_client as polygon_api
import requests

class LLMStockTrader:
    def __init__(self, starting_balance: int, max_trades_per_day=3):
        # Portfolio
        self.cash_balance = starting_balance
        self.holdings = {"AAPL": 32}
        self.trade_history = ["BUY AAPL 20", "BUY AAPL 12"]
        self.action_summaries = ["Positive market sentiment, upward trend, and potential for growth driven by technology sector gains."]

        self.max_trades_per_day = max_trades_per_day
        self.daily_trades_executed = 3

        self.llm_url = f"http://localhost:5000{os.getenv('BASE_PATH')}/chat"
        self.llm_auth = (os.getenv("APP_USER"), os.getenv('APP_PASS'))


    def observe(
            self, ticker: str, now: pd.Timestamp,
            hourly_lookback_days: int, daily_bars: int, 
            weekly_bars: int, monthly_bars: int, news_articles: int,
            trade_history: int, action_summaries: int) -> tuple[str, float]:
        """
        Step 1: Fetch live prices and news
        """

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
            df_hour = df_hour.reset_index()

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
        - Current Price Per Share: {current_price}
        - Current Portfolio Cash: ${self.cash_balance:.2f}
        - Current Holdings (Shares): {self.holdings}
        - Trade History: {self.trade_history[-trade_history:] if len(self.trade_history) >= trade_history else self.trade_history}
        - Trades Executed Today: {self.daily_trades_executed}
        - Previous Action Summaries: {self.action_summaries[-action_summaries:] if len(self.action_summaries) >= action_summaries else self.action_summaries}
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
    

    def reason(self, mkt_context: str, max_shares_per_trade:int) -> dict:
        """
        Step 2: Make decision using LLMs
        """

        print("[REASON] Analyzing market using DeepSeek R1...")


        system_prompt = (
            "You are an elite, purely logical quantitative AI used at a large firm. Your Goal: Maximize Returns while adhering to a medium-risk strategy! "
            "You have access to the market and are able to execute trades based on cash position and portfolio. "
            "Analyze the provided market data, news, and portfolio content to make investment decisions for the given ticker. "
            "For your response, you must output a single, valid JSON object containing your final decision. "
            "The JSON must strictly match this format (use double quotes): "
            '{"action": "BUY | SELL | HOLD | ALERT", "ticker": "STRING | null", "shares": "INTEGER | null", "rationale": "STRING"} '
            f"You may BUY or SELL up to {max_shares_per_trade} shares in a single transaction based on portfolio constraints. "
            f"You have the ability to make decisions every hour while the market is open. However, you are allowed a maximum of {self.max_trades_per_day} trades per day. "
            "Your 'rationale' field should be a 1-2 sentence justification to guide your future decisions. Discuss what you learned and justify your decision. "
            "After your <think> reasoning process, output ONLY the valid JSON object. Do not wrap it in markdown formatting or provide conversational text. "
            "In case of issues, use the ALERT action to communicate anything urgent. Good Luck! "
        )

        payload = {
            "system_prompt": system_prompt,
            "prompt": mkt_context,
            "temperature": 0.1
        }

        # Invoke the local LLM docker container
        try:
            response = requests.post(
                self.llm_url,
                json=payload,
                auth=self.llm_auth,
                timeout=float(os.getenv("LLM_TIMEOUT"))
            )

            response.raise_for_status()
            response_json = response.json()

            message = response_json["choices"][0]["message"]

            final_content = message["content"].strip()
            reasoning = message["reasoning_content"].strip()
            usage = response_json["usage"]

            print("[REASON] Success! Recieved decision from LLM...")

            return {
                "content": final_content,
                "reasoning": reasoning,
                "usage": usage
            }
        
        except requests.exceptions.RequestException as e:
            print(f"Error invoking LLM; exception={e}")
            raise e