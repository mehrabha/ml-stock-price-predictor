import os
import pandas as pd
import pandas_market_calendars as mkt_cal
import src.api_client as polygon_api
import json, requests

class LLMStockTrader:
    def __init__(self, starting_balance: int):
        # Portfolio
        self.cash_balance = starting_balance
        self.holdings = {}
        self.portfolio_val = starting_balance
        self.trade_history = []
        self.action_summaries_week = []
        self.action_summaries_month = []

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
            raise ValueError("[OBSERVE] Stock Market is closed today!")
    
        # go back X trading days
        hourly_start = mkt_days[-(hourly_lookback_days + 1)].strftime("%Y-%m-%d")
        daily_start = mkt_days[-(daily_bars + 2)].strftime("%Y-%m-%d")
        weekly_start = (now - pd.DateOffset(weeks=weekly_bars + 2)).strftime("%Y-%m-%d")
        monthly_start = (now - pd.DateOffset(months=monthly_bars + 2)).strftime("%Y-%m-%d")

        try:
            df_hour = polygon_api.get_prices(ticker, "hour", hourly_start, end_date)
            df_day = polygon_api.get_prices(ticker, "day", daily_start, end_date)
            df_week = polygon_api.get_prices(ticker, "week", weekly_start, end_date)
            df_month = polygon_api.get_prices(ticker, "month", monthly_start, end_date)
            df_news = polygon_api.get_news(ticker, end_date, end_date)
        except Exception as e:
            raise Exception(f"[OBSERVE] Error invoking price APIs; cause:{e}")
        
        # trim extra hours 
        if df_hour is not None and not df_hour.empty:
            df_hour = df_hour.set_index("eastern_dt")
            df_hour = df_hour.between_time("8:00", "18:00", inclusive="left")
            df_hour = df_hour.reset_index()

        # prevent lookahead bias while validating against historical prices
        df_hour = df_hour[df_hour["eastern_dt"] < now.floor('h')]
        df_news = df_news[df_news["eastern_dt"].dt.hour.between(now.hour - 4, now.hour)]
        df_day = df_day.iloc[:-1]
        df_week = df_week.iloc[:-1]
        df_month = df_month.iloc[:-1]

        if df_news is not None and not df_news.empty:
            df_news = df_news[["eastern_dt", "title", "publisher", "description"]].sort_values(by="eastern_dt")
            md_news = df_news.head(news_articles).to_markdown(index=False)
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
            raise ValueError(f"[OBSERVE] Insufficient data for {ticker}; API returned fewer price points...")

        # latest price
        current_price = df_hour["close"].iloc[-1]
        self.portfolio_val = self.cash_balance
        self.portfolio_val += self.holdings[ticker] * current_price if ticker in self.holdings else 0   # TODO support multi stock portfolios

        # construct the final context payload
        context = f"""
        **Current Market State for {ticker}**
        - Current Price Per Share: {current_price}
        - Current Portfolio Cash: ${self.cash_balance:.2f}
        - Current Holdings (Shares): {self.holdings}
        - Current Portfolio Value: ${self.portfolio_val}
        - Trade History: {self.trade_history[-trade_history:] if len(self.trade_history) >= trade_history else self.trade_history}
        - Previous Action Summaries (Week): {self.action_summaries_week[-action_summaries:] if len(self.action_summaries_week) >= action_summaries else self.action_summaries_week}
        - Previous Action Summaries (Month): {self.action_summaries_month[-action_summaries:] if len(self.action_summaries_month) >= action_summaries else self.action_summaries_month}
        - Clock: {now}

        ### Recent News (Caution! News may be missing or irrelevant)
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
        
        print("[OBSERVE] Success!\n")
        return context, current_price
    

    def reason(self, ticker: str, now: pd.Timestamp, mkt_context: str, max_shares_per_trade:int) -> dict:
        """
        Step 2: Make decision using LLMs
        """

        print("[REASON] Analyzing market using DeepSeek R1...")


        system_prompt = (
            "You are an elite, purely logical quantitative AI used at a large firm. Your Goal: Maximize Returns while adhering to a medium-risk strategy! "
            "You have access to the market and are able to execute trades based on cash position and portfolio. "
            "Analyze the provided market data, news, portfolio content and summaries to make investment decisions for the given ticker. "
            f"For your response, you must output a single, valid JSON object containing your final decision regarding ticker={ticker}. "
            "The JSON must strictly match this format (use double quotes): "
            '{"action": "BUY | SELL | HOLD | ALERT", "ticker": "STRING | null", "shares": "INTEGER | null", "rationale": "STRING"} '
            f"You may BUY or SELL up to {max_shares_per_trade} shares in a single transaction based on portfolio constraints. "
            f"You have the ability to make two decisions per day. The execution loop runs every business day at 10:00, 14:00 and 18:00. "
            "Don't execute BUY/SELL during after-hours (post 18:00). Instead, use the time to analyze news articles and key events which can be used to guide future decisions. "
            "Your 'rationale' field should be a 1-3 sentence justification to guide your future decisions. Discuss prices/news articles to justify your decision. "
            "IMPORTANT: Output ONLY the valid JSON object. Do not wrap it in markdown formatting, extra tags (eg. ```json) or provide conversational text. "
            "Use the ALERT action to communicate technical/data issues ONLY. Good Luck! "
        )

        payload = {
            "system_prompt": system_prompt,
            "prompt": f"current_timestamp: {now}, mkt_context: {mkt_context}",
            "temperature": 0.1,
            "cache_prompt": False
        }

        # Invoke the local LLM docker container
        try:
            response = requests.post(
                self.llm_url,
                json=payload,
                auth=self.llm_auth,
                timeout=float(os.getenv("LLM_TIMEOUT")),
            )

            response.raise_for_status()
            response_json = response.json()

            missing = []
            for field in ["content", "reasoning", "usage"]:
                if field not in response_json:
                    missing.append(field)

            if len(missing) > 0:
                raise Exception(f"[REASON] Required fields missing from LLM response: {missing}")
            
            print("[REASON] Success! Recieved decision from LLM...")
            print(f"[REASON] Metrics: {response_json['usage']}\n")
            return response_json
        
        except requests.exceptions.RequestException as e:
            print(f"[REASON] Error invoking LLM; exception={e}")
            raise e
    
    def plan(self, tk: str, now: pd.Timestamp, llm_outout_str: str, current_price: float, max_shares_per_trade: int) -> tuple[str, str, int, str]:
        """
        Step 3: Parse the LLM's JSON output, apply guardrails and determine whether the trade is valid
        """

        print("[PLAN] Processing LLM decision against portfolio...")

        # parse the llm output
        try:
            decision = json.loads(llm_outout_str.strip())
            action = decision["action"]
            ticker = decision["ticker"]
            shares = decision["shares"] if action in ["BUY", "SELL"] else 0
            rationale = decision.get("rationale")
        except json.JSONDecodeError as e:
            raise Exception("[PLAN] LLM failed to output valid JSON")
        except KeyError as e:
            raise Exception(f"[PLAN] Key missing in LLM output: {e}")
        except Exception as e:
            raise Exception(f"[PLAN] Error: {e.with_traceback}")


        # apply guardrails
        if ticker != tk:
            raise Exception(f"[PLAN] Invalid ticker; expected={tk}; provided={ticker}")
        if action == "ALERT":
            print(f"[PLAN] Warning! LLM sent an ALERT! {action} {rationale}")
            return "HOLD", ticker, shares, rationale
        if action not in ["BUY", "SELL", "HOLD"]:
            raise Exception(f"[PLAN] Invalid action by Agent: {action}")
        if action in ["BUY", "SELL"] and (not shares or shares <= 0):
            raise Exception(f"[PLAN] Shares qty is 0/None for BUY/SELL actions")
        

        # Force a HOLD for unapproved trades
        if now.hour not in range(9, 16):
            return "HOLD", ticker, shares, f"After hours trading detected, hour={now.hour}; Forcing a HOLD!"
        
        if shares > max_shares_per_trade:
            return "HOLD", ticker, shares, f"Share qty exceeded limit={max_shares_per_trade}; Forcing a HOLD!"
        
        if action == "BUY" and shares * current_price > self.cash_balance:
            return "HOLD", ticker, shares, f"Insufficient balance={self.cash_balance} to execute {action} {ticker} {shares}; Forcing a HOLD!"
        
        if action == "SELL" and (ticker not in self.holdings or shares > self.holdings[ticker]):
            return "HOLD", ticker, shares, f"Unable to execute {action} {ticker} {shares}; current holdings={self.holdings}; Forcing a HOLD!"

        print(f"[PLAN] Trade approved!! {action} {ticker} {shares} {rationale} \n")

        return action, ticker, shares, rationale


    def execute(self, action: str, ticker: str, now: pd.Timestamp, qty: int, current_price: float, rationale: str) -> float:
        """
        Step 4: Execute the trade by invoking exchange API and update portfolio state.
        """

        if action not in ["BUY", "SELL", "HOLD"]:
            raise Exception(f"[EXECUTE] Cannot execute trade; invalid action={action}")
        
        # TODO API call to exchange
        if action in ["BUY", "SELL"]:
            print(f"[EXECUTE] Invoking broker API for {action} {qty} {ticker} (Mocked atm!)...")
        call_failed = False

        if call_failed:
            raise Exception("[EXECUTE] Broker rejected the trade...")
        
        if action == "BUY":
            prix = qty * current_price
            self.cash_balance -= prix
            self.holdings[ticker] = self.holdings.get(ticker, 0) + qty
        elif action == "SELL":
            prix = qty * current_price
            self.cash_balance += prix
            self.holdings[ticker] = self.holdings.get(ticker, 0) - qty
            if self.holdings[ticker] == 0:
                self.holdings.pop(ticker)

        # recalculate portfolio
        self.portfolio_val = self.cash_balance
        self.portfolio_val += self.holdings[ticker] * current_price if ticker in self.holdings else 0

        self.trade_history.append(f"{now.strftime('%Y-%m-%d')} portfolio_value:{self.portfolio_val}; '{action} {ticker} {qty}' @ {current_price:.2f}; rationale:'{rationale}'")
        
        print(f"[EXECUTE] Success! New balance: ${self.cash_balance:.2f} | Holdings: {self.holdings} \n")
        return self.portfolio_val
    

    def reflect(self, ticker:str, now: pd.Timestamp) -> tuple[str, str]:
        """
        Step 5: Invoke LLM, compresses trade history into high-level summaries.
        """

        if now.weekday() != 4:
            raise Exception("[REFLECT] this method should only be invoked on Fridays!")

        is_last_friday = (now + pd.Timedelta(days=7)).month != now.month    # generate month end summary

        if not self.trade_history and not is_last_friday:
            week_summary = f"Summary for Week Ending in {now.strftime('%Y-%m-%d')}: No trades were executed!"
            self.action_summaries_week.append(week_summary)
            return week_summary, "No trades to summarize..."
        elif not self.trade_history and not self.action_summaries_week:
            month_summary = f"Summary for Month {now.strftime('%Y-%m-%d')}: No trades were executed!"
            self.action_summaries_month.append(month_summary)
            return month_summary, "No trades to summarize..."
        elif not is_last_friday:
            print(f"[REFLECT] Initiating reflection for Week Ending in {now.strftime('%Y-%m-%d')}...")
            print(f"[REFLECT] Sumarizing {len(self.trade_history)} trades...")
            history_text = '\n'.join(self.trade_history)

            system_prompt = (
                "You are an elite quantitative AI reflecting on your weekly trading performance. "
                "Review the provided trade ledger and output a 3-6 sentence summary of the trading week. "
                "Reflect on the market and summarize what you've learned. Did you make any decisions to BUY or SELL?"
                f"For your response, you must output a single, valid JSON object containing your final analysis for ticker={ticker} as 'message'. "
                "The JSON must strictly match this format (use double quotes): "
                '{"action": "REFLECT | ALERT", "ticker": "STRING | null", "message": "STRING"} '
                "IMPORTANT: Output ONLY the valid JSON object. Do not wrap it in markdown formatting, extra tags (eg. ```json) or provide conversational text. "
                "Use the ALERT action to communicate technical/data issues ONLY with a message. Good Luck! "
            )

            print(f"[REFLECT] Market closed for the weekend. Initiating reflection for {ticker}...")

            payload = {
                "system_prompt": system_prompt,
                "prompt": f"Trading Ledger:\n{history_text}",
                "temperature": 0.1,
                "cache_prompt": False
            }

            try:
                response = requests.post(
                    self.llm_url,
                    json=payload,
                    auth=self.llm_auth,
                    timeout=float(os.getenv("LLM_TIMEOUT"))
                )

                response.raise_for_status()
                response_json = response.json()

                missing = []
                for field in ["content", "reasoning", "usage"]:
                    if field not in response_json:
                        missing.append(field)

                if len(missing) > 0:
                    raise Exception(f"[REFLECT] Required fields missing from LLM response: {missing}")
                
                parsed_json = json.loads(response_json["content"].strip())
                action = parsed_json["action"]
                message = parsed_json["message"]

                if action == "ALERT":
                    print(f"[REFLECT] Warning! LLM sent an ALERT! {action} {message}")
                    return action, message
                
                print("[REFLECT] Success! Recieved summary for week...")
                print(f"[REFLECT] Metrics: {response_json['usage']}\n")

                self.action_summaries_week.append(message)
                self.trade_history.clear()
                return message, response_json["reasoning"]
            except requests.exceptions.RequestException as e:
                raise Exception(f"[REFLECT] Error invoking LLM; exception={e}")
        else:
            print(f"[REFLECT] Initiating reflection for Month End {now.strftime('%Y-%m')}...")

            print(f"[REFLECT] Summarizing {len(self.trade_history)} trades...")
            history_text = '\n'.join(self.trade_history)
            weeky_summaries_txt = '\n'.join(self.action_summaries_week)

            print(f"[REFLECT] Summarizing {len(self.action_summaries_week)} summaries from prev weeks...")

            system_prompt = (
                "You are an elite quantitative AI reflecting on your monthly trading performance. "
                "Review the provided trade ledger + weekly summaries and output 5-7 sentences describing the trading month. "
                "Analyze the portfolio performance and see how it's doing. Did you make any gains? What went wrong? "
                "Reflect on the market and your actions. Identify key wins/mistakes. " 
                "How would you adjust your trading strategy going forward? Leave a memo for your future self "
                f"For your response, you must output a single, valid JSON object containing your final analysis for ticker={ticker} as 'message'. "
                "The JSON must strictly match this format (use double quotes): "
                '{"action": "REFLECT | ALERT", "ticker": "STRING | null", "message": "STRING"} '
                "IMPORTANT: Output ONLY the valid JSON object. Do not wrap it in markdown formatting, extra tags (eg. ```json) or provide conversational text. "
                "Use the ALERT action to communicate technical/data issues ONLY with a message. Good Luck! "
            )

            print(f"[REFLECT] Market closed for the weekend. Initiating reflection for {ticker}...")

            payload = {
                "system_prompt": system_prompt,
                "prompt": f"Trading Ledger:\n{history_text}\nWeekly Summaries:\n{weeky_summaries_txt}",
                "temperature": 0.1,
                "cache_prompt": False
            }

            try:
                response = requests.post(
                    self.llm_url,
                    json=payload,
                    auth=self.llm_auth,
                    timeout=float(os.getenv("LLM_TIMEOUT"))
                )

                response.raise_for_status()
                response_json = response.json()

                missing = []
                for field in ["content", "reasoning", "usage"]:
                    if field not in response_json:
                        missing.append(field)

                if len(missing) > 0:
                    raise Exception(f"[REFLECT] Required fields missing from LLM response: {missing}")
                
                parsed_json = json.loads(response_json["content"].strip())
                action = parsed_json["action"]
                message = parsed_json["message"]

                if action == "ALERT":
                    print(f"[REFLECT] Warning! LLM sent an ALERT! {action} {message}")
                    return action, message
                
                print("[REFLECT] Success! Recieved summary for month...")
                print(f"[REFLECT] Metrics: {response_json['usage']} \n")

                self.action_summaries_month.append(message)
                self.trade_history.clear()
                self.action_summaries_week.clear()
                return message, response_json["reasoning"]
            except requests.exceptions.RequestException as e:
                print(f"[REFLECT] Error invoking LLM; exception={e}")
                raise e
        
                
            