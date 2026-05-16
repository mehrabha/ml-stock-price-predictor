
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader

from src.dataset import Dataset
from src.model_dataset import AlphaTraderDataset
from src.model import AlphaTrader
from src.train import Trainer


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
    start_dt = datetime(2024, 6, 1)
    end_dt = datetime(2026, 4, 30)
    d = Dataset(tk, start_dt, end_dt)

    # fetch stock prices from polygion.io
    d.download_csvs()
    time.sleep(2)

    # parse csvs into training
    # following params should be set based on available data, otherwise will result in index wrap arounds/exceptions
    train_price_predictor(d, 
        training_start_dt=datetime(2024, 10, 1),
        training_end_dt=datetime(2025, 10, 31),
        val_start_dt=datetime(2025, 11, 1),
        val_end_dt=datetime(2026, 4, 30),
        hourly_lookback_days=5,
        daily_bars=21, 
        weekly_bars=13, 
        monthly_bars=3, 
        max_news_per_hr=3
    )

    

def train_price_predictor(d, training_start_dt, training_end_dt,
                          val_start_dt, val_end_dt,
                          hourly_lookback_days, daily_bars, weekly_bars, monthly_bars, max_news_per_hr):
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
