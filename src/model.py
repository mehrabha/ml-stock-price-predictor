import torch
import torch.nn as nn

class AlphaTrader(nn.Module):
    def __init__(self, hourly_bars: int, macro_bars: int, news_blocks: int, extra_params: int):
        super(AlphaTrader, self).__init__()
        self.hourly_bars = hourly_bars
        self.macro_bars = macro_bars
        self.news_blocks = news_blocks
        self.extra_params = extra_params

        # BRANCH 1: Convolution layer for analyzing prices
        self.price_analysis = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=32, kernel_size=3, padding=1),    # convolution based on 3 hr blocks
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2),    # look at every two hour window, reducing overfitting
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),   # deep feature extractions
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * hourly_bars // 2, 64)  # compress to a dense array of 64 numbers
        )

        # BRANCH 2: Capture long term prices
        self.macro_trends = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * macro_bars, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64)  # Compress to match the prices branch
        )

        # BRANCH 3: Transformer encoder block for analyzing news sequence
        self.news_transformer = nn.Sequential(
            nn.Linear(386, 128),    # Project raw features into 128 dimentional embedding
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),     # multi head attention layer
            nn.Flatten(),
            nn.Linear(128 * news_blocks, 64)  # Compress to match the price branch
        )

        # FUSION LAYER: Linear classifier based on inputs from price, macro and news layers
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64 + extra_params, 64), # outputs plus handful of useful params eg. weekday, day, month etc...
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 1)
        )
    
    def forward(self, hours: torch.Tensor, macros: torch.Tensor, news: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        invalid = (
            hours.size(1) != 7 or                       # Checks feature depth (Open, High, Low...)
            hours.size(2) != self.hourly_bars or        # Checks timeline length (47)
            macros.size(1) != 7 * self.macro_bars or    # Checks macro array size
            news.size(1) != self.news_blocks or         # Checks timeline length (8)
            news.size(2) != 386 or                      # Checks semantic embedding depth
            params.size(1) != self.extra_params         # Checks extra params count
        )

        if invalid:
            raise ValueError("Training row has unexpected number of elements!")
        
        # Evaluate the arrays
        p_out = self.price_analysis(hours)
        m_out = self.macro_trends(macros)
        n_out = self.news_transformer(news)

        combined = torch.cat((p_out, m_out, n_out, params), dim=1)

        # get final output
        return self.fusion(combined)
    
    