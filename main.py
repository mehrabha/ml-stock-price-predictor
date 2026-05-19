
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import pandas_market_calendars as mkt_cal
import numpy as np
import time, traceback
import json
from torch.utils.data import DataLoader

from src.dataset import Dataset
from src.model_dataset import AlphaTraderDataset
from src.model import AlphaTrader
from src.train import Trainer
from src.llm_trader import LLMStockTrader


load_dotenv()


# semantics analysis: convert news articles to vectors for training
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_semantics(row):
    # iterate over hours
    for timeframe in row:
        samples = timeframe["samples"]
        timeframe["samples"] = len(samples) # preserve the sample count

        if len(samples) == 0:
            timeframe["semantics"] = np.zeros(384).tolist()  #No news available
        else:
            # select fields to encode
            text_to_encode = [f"""
                              title={sample["title"]}-
                              publisher={sample["publisher"]}-
                              author={sample["author"]}-
                              description={sample["description"]}-
                              keywords={sample["keywords"]}-
                              tickers={sample["tickers"]}
                              """ for sample in samples]
            
            semantics = model.encode(text_to_encode)    # mean pooling
            timeframe["semantics"] = np.mean(semantics, axis=0).tolist()
        
    
    return row


def main():
    # --- 1. DATA DOWNLOAD PIPELINE ---
    tk = "AAPL"
    start_dt = datetime(2025, 1, 1)
    end_dt = datetime(2025, 12, 31)
    #d = Dataset(tk, start_dt, end_dt)

    # fetch stock prices from polygion.io
    #d.download_csvs()
    #time.sleep(2)

    # initialize llm trading agent
    ticker = "AAPL"
    agent = LLMStockTrader(starting_balance=10000.00)

    validate_llm_agent(agent, ticker, start_dt, end_dt)



def validate_llm_agent(agent, ticker, start_date, end_date, starting_balance=10000.00, shares_per_trade=10):
    portfolio_values = [starting_balance]
    agent_actions = {}
    exceptions = []

    baseline_values = [starting_balance]
    baseline_cash = starting_balance
    baseline_stocks = 0

    # get valid nyse market days
    mkt_days = mkt_cal.get_calendar("NYSE").valid_days(start_date, end_date)
    mkt_days = mkt_days.tz_localize(None).tz_localize("US/Eastern").to_list()

    print(f"Validation Begin : Test Window has {len(mkt_days)} days")
    
    trade_simulations = []

    for date in mkt_days:
        trade_simulations.append(date.replace(hour=12)) # 12 AM Trades
        trade_simulations.append(date.replace(hour=18)) # After hour research/reflections

    for trade_simulation_dt in trade_simulations:
        print(f"..........Simulating trade for time={trade_simulation_dt} : BEGIN..........")

        try:
            # STEP 1: Fetch prices and news articles
            context, current_price = agent.observe(
                ticker=ticker,
                now = trade_simulation_dt,
                hourly_lookback_days=1,
                daily_bars=10,
                weekly_bars=6,
                monthly_bars=4,
                news_articles=6,
                trade_history=20,
                action_summaries=5,
                baseline_metrics=f"Buy & Hold; portfolio value=${baseline_values[-1]:.2f}"
            )

            print(context + '\n\n')
            #time.sleep(1)

            # STEP 2: Invoke LLM for a trading decision
            decision = agent.reason(
                ticker=ticker,
                now=trade_simulation_dt,
                mkt_context=context,
                max_shares_per_trade=shares_per_trade
            )
            #time.sleep(1)

            print(f"LLM Chain of Thoughts: {decision['reasoning'][:300]}...", end='\n\n')

            # STEP 3: Validate Trade and prepare to publish
            action, ticker, shares, rationale = agent.plan(
                tk=ticker,
                now=trade_simulation_dt,
                llm_outout_str=decision["content"],
                current_price=current_price,
                max_shares_per_trade=shares_per_trade
            )

            print(f"Decision: {action}\nticker: {ticker}\nshares: {shares}\nRationale: {rationale}...", end='\n\n')

            if action in agent_actions:
                agent_actions[action] += 1
            else:
                agent_actions[action] = 1

            #time.sleep(1)

            # STEP 4: Execute Trade
            portfolio_value = agent.execute(
                action=action,
                ticker=ticker,
                now=trade_simulation_dt,
                qty=shares, 
                current_price=current_price,
                rationale=rationale
            )

            portfolio_values.append(portfolio_value)

            # calculate baseline, buys 1 share at a time if possible
            if current_price <= baseline_cash and trade_simulation_dt.hour < 16:
                baseline_cash -= current_price
                baseline_stocks += 1

            baseline_value = baseline_cash + baseline_stocks * current_price
            baseline_values.append(baseline_value)
            #time.sleep(1)

            # STEP 5: Reflection/Summarize on Fridays
            if trade_simulation_dt.weekday() == 4 and trade_simulation_dt.hour >= 16:
                msg, reasoning = agent.reflect(
                    ticker=ticker,
                    now=trade_simulation_dt
                )

                print(f"Reflection: {msg}\n\nTicker: {ticker}\n\nReasoning: {reasoning[:500]}...", end='\n\n')

        except Exception as e:
            exception_txt = traceback.format_exc()
            print(f"Exception occured during execution loop for {trade_simulation_dt}; {exception_txt}..........")
            
            exceptions.append(exception_txt)
            print("Continuing...")
        finally:
            print(f"Loop Completed!\nActions: {agent_actions};\nPortfolio Value: {portfolio_values[-1]}\nBaseline Value: {baseline_values[-1]}\nErrors: {len(exceptions)}")

            validation_results = {
                "portfolio_final": portfolio_values[-1],
                "baseline_final": baseline_values[-1],
                "test_duration": str(len(mkt_days)) + " days",
                "actions": agent_actions,
                "exceptions_count": len(exceptions),
                "exceptions": exceptions,
                "portfolio_trend": portfolio_values,
                "baseline_trend": baseline_values,
            }

            with open("validation_results.json", 'w') as f:
                json.dump(validation_results, f, indent=4)
            
            print(f"..........{trade_simulation_dt} : END..........\n")
            #time.sleep(2)

    print("Validation End : Successfully persisted validation results!")


def price_predictor(d, training_start_dt, training_end_dt,
                    val_start_dt, val_end_dt,
                    hourly_lookback_days, daily_bars, weekly_bars, monthly_bars, max_news_per_hr):
    

    # train and validate price predictor using csvs
    # following params should be set based on available data, otherwise will result in index wrap arounds/exceptions
    # price_predictor(d, 
    #     training_start_dt=datetime(2024, 9, 1),
    #     training_end_dt=datetime(2025, 8, 31),
    #     val_start_dt=datetime(2025, 9, 1),
    #     val_end_dt=datetime(2026, 4, 30),
    #     hourly_lookback_days=5,
    #     daily_bars=21, 
    #     weekly_bars=13, 
    #     monthly_bars=3, 
    #     max_news_per_hr=3
    # )
    
    # --- 2. PREPARE TRAINING SET ---
    train_label = d.generate_training_dataset(
        training_start_dt,
        training_end_dt, 
        hourly_lookback_days, 
        daily_bars, 
        weekly_bars, 
        monthly_bars, 
        max_news_per_hr,
        label="training_set",
        normalize_datasets=True
    )
    # similarly create test data, look back days must match!
    test_label = d.generate_training_dataset(
        val_start_dt,
        val_end_dt, 
        hourly_lookback_days, 
        daily_bars, 
        weekly_bars, 
        monthly_bars, 
        max_news_per_hr,
        label="testing_set",
        normalize_datasets=True
    )

    training_set = pd.read_parquet(f"data/processed/{train_label}.parquet")
    validation_set = pd.read_parquet(f"data/processed/{test_label}.parquet")

    print(f"\nTraining Set:\n{training_set.iloc[0]}")
    print(f"\nValidation Set:\n{validation_set.iloc[0]}")
    time.sleep(1)

    # --- 3. SEMANTICS ANALYSIS ---
    print("\nEncoding semantics... this may take a moment...")
    training_set["news"] = training_set["news"].apply(get_semantics)
    file = f"./data/processed/{train_label}_with_semantics.parquet"
    training_set.to_parquet(path=file, engine="pyarrow")
    print(f"Success! file:{file}, rows:{len(training_set)}")

    print("Encoding semantics for test data... this may take a moment...")
    validation_set["news"] = validation_set["news"].apply(get_semantics)
    file2 = f"./data/processed/{test_label}_with_semantics.parquet"
    validation_set.to_parquet(path=file2, engine="pyarrow")
    print(f"Success! file:{file2}, rows:{len(validation_set)}")

    time.sleep(2)

    # --- 4. PYTORCH DEEP LEARNING ---
    print("\n\nInitializing PyTorch DataLoaders...")

    train_data = AlphaTraderDataset(file)
    val_data = AlphaTraderDataset(file2)

    # Batch size 16 is a great sweet spot for multimodal models
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    time.sleep(1)
    print("\n\nInitializing Neural Nets...")

    model = AlphaTrader(
        hourly_bars=hourly_lookback_days * 10 + 7,          # 10 bars per day + 7 bars from today
        macro_bars=daily_bars + weekly_bars + monthly_bars, # 21 + 13 + 3
        news_blocks=8,                                      # 3-5pm, 5-7pm, 7am-11am, 11-1pm, 1-3pm, saturday, sunday (sorted),
        extra_params=4                                      # integers: weekday, day, month, quarter
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda"
    )

    time.sleep(1)

    # --- 5. LIFT OFF! ---
    print("\n\n Commencing Training...")
    trainer.fit(epochs=20, tag="weights.pth")


if __name__ == "__main__":
    main()
