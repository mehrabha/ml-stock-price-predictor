from src.dataset import Dataset
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv()

# create dataset
tk = "AAPL"
start_dt = datetime(2024, 6, 1)
end_dt = datetime(2026, 3, 31)

# initialize dataset class with date range
d = Dataset(tk, start_dt, end_dt)

# fetch stock prices from polygion.io
d.download_csvs()

time.sleep(2)

training_start_dt = datetime(2024, 10, 1)
training_end_dt = datetime(2025, 9, 30)

# following params should be set based on available data, otherwise will result in index wrap arounds/exceptions
d.generate_training_dataset(
    training_start_dt,
    training_end_dt, 
    hourly_lookback_days=4, 
    daily_bars=14, 
    weekly_bars=10, 
    monthly_bars=4, 
    max_news_per_hr=3,
    label="training_set"
    )