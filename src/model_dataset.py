from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class AlphaTraderDataset(Dataset):
    def __init__(self, parquet_file_path):
        self.df = pd.read_parquet(parquet_file_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get specific rows for pytorch (e.g. row 42)
        row = self.df.iloc[idx]

        # Extract features
        features = ["open", "high", "low", "close", "volume", "vwap", "transactions"]

        prices = []
        for price_bar in row["hour"]:
            points = [price_bar[feature] for feature in features]
            prices.append(points)

        macros = []
        for bar_type in ["day", "week", "month"]:
            for price_bar in row[bar_type]:
                for feature in features:
                    macros.append(price_bar[feature])

        news_semantics = []
        for news in row["news"]:    
            # there should be 8 of them, save the embeddings + extra info as 386d vectors
            semantics = news["semantics"]
            semantics.append(news["count"])     
            semantics.append(news["samples"])
            news_semantics.append(semantics)

        #convert to numpy/tensors
        prices_mtx = np.array(prices) 
        macros_array = np.array(macros)
        news_mtx = np.array(news_semantics)
        params_array = np.array(row["params"])
        target_val = row['target']  # 0 or 1

        # 5. Package
        return {
            'hours': torch.tensor(prices_mtx, dtype=torch.float32).transpose(0, 1),     # transpose to 7x57
            'macros': torch.tensor(macros_array, dtype=torch.float32),
            'news': torch.tensor(news_mtx, dtype=torch.float32),
            'params': torch.tensor(params_array, dtype=torch.float32),
            'target': torch.tensor(target_val, dtype=torch.float32)
        }