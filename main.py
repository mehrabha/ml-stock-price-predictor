from src.dataset import Dataset
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# create dataset
tk = "AAPL"
start_dt = datetime(2024, 6, 1)
end_dt = datetime(2026, 3, 31)

# initialize dataset class with date range
d = Dataset(tk, start_dt, end_dt)

# fetch stock prices from polygion.io
d.get_csvs()