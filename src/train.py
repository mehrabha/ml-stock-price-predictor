import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model: nn.Module, train_loader:DataLoader, val_loader: DataLoader, device: str = "cuda"):
        print("Initializing trainer...")
        if not torch.cuda.is_available():
            print("Warning! NVIDIA GPU not detected, defaulting to CPU...")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss: Much more mathematically stable than doing Sigmoid + BCELoss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.best_val_loss = float("inf")

        print(f"Trainer initialized, device: {self.device}")
    def train_epoch(self) -> float:
        self.model.train
