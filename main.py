from src.dataset import Dataset
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import time

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
    # initialize dataset class and create dataset
    tk = "AAPL"
    start_dt = datetime(2024, 6, 1)
    end_dt = datetime(2026, 4, 30)
    d = Dataset(tk, start_dt, end_dt)

    # fetch stock prices from polygion.io
    d.download_csvs()

    time.sleep(2)

    training_start_dt = datetime(2024, 11, 1)
    training_end_dt = datetime(2025, 10, 31)

    # parse csvs into training
    # following params should be set based on available data, otherwise will result in index wrap arounds/exceptions
    label = d.generate_training_dataset(
        training_start_dt,
        training_end_dt, 
        hourly_lookback_days=4, 
        daily_bars=30, 
        weekly_bars=10, 
        monthly_bars=4, 
        max_news_per_hr=3,
        label="training_set"
    )

    training_set = pd.read_parquet(f"data/processed/{label}.parquet")

    print(training_set.iloc[0])

    print("Encoding semantics... this may take a moment.")
    training_set["news"] = training_set["news"].apply(get_semantics)

    file = f"./data/processed/{label}_with_semantics.parquet"
    training_set.to_parquet(path=file, engine="pyarrow")
    print(f"Success! {file}")

    
if __name__ == "__main__":
    main()
