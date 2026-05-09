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

        print(f"Trainer initialized! GPU: {self.device}")

        
    def train(self) -> float:
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        baseline_correct = 0    # buy and hold strategy: count the number of up days
        total_predictions = 0

        fancy_visuals = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in fancy_visuals:
            # 1. Load all tensors into the GPU
            hours = batch['hours'].to(self.device, dtype=torch.float32)
            macros = batch['macros'].to(self.device, dtype=torch.float32)
            news = batch['news'].to(self.device, dtype=torch.float32)
            params = batch['params'].to(self.device, dtype=torch.float32)
            targets = batch['target'].to(self.device, dtype=torch.float32).unsqueeze(1)


            # 2. Zero gradients
            self.optimizer.zero_grad()

            # 3. Forward pass
            logits = self.model(hours, macros, news, params)

            # 4. Compute Loss
            loss = self.criterion(logits, targets)

            # 5. Backward Pass
            loss.backward()

            # 6. Gradient Clipping
            # Keeps the attention heads from exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 7. Update Weights
            self.optimizer.step()

            # 8. Convert outputs to 0/1s
            running_loss += loss.item()
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            baseline_correct += (1 == targets).sum().item()
            total_predictions += targets.size(0)

            fancy_visuals.set_postfix(loss=f"{loss.item():.4f}:")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_predictions
        baseline_acc = baseline_correct / total_predictions
        return epoch_loss, epoch_acc, baseline_acc
    
    def validate(self) -> float:
        self.model.eval()   # Put model in evaluation mode
        val_loss = 0.00
        correct = 0
        baseline_correct = 0    # buy and hold strategy: count the number of up days
        total_predictions = 0

        with torch.no_grad():   # evaludate test set without gradient descent/backpropagation
            for batch in self.val_loader:
                hours = batch['hours'].to(self.device, dtype=torch.float32)
                macros = batch['macros'].to(self.device, dtype=torch.float32)
                news = batch['news'].to(self.device, dtype=torch.float32)
                params = batch['params'].to(self.device, dtype=torch.float32)
                targets = batch['target'].to(self.device, dtype=torch.float32).unsqueeze(1)

                logits = self.model(hours, macros, news, params)
                loss = self.criterion(logits, targets)

                val_loss += loss.item()
                predictions = (torch.sigmoid(logits) >= .5).float()
                correct += (predictions == targets).sum().item()
                baseline_correct += (1 == targets).sum().item()
                total_predictions += targets.size(0)
        
            epoch_loss = val_loss / len(self.val_loader)
            epoch_acc = correct / total_predictions
            baseline_acc = baseline_correct / total_predictions
            return epoch_loss, epoch_acc, baseline_acc
        

    def fit(self, epochs: int, save_path: str = "alpha_trader.pth"):
        print(f"Starting training on device: {self.device}")
        
        for epoch in range(epochs):
            train_loss, train_acc, train_base = self.train()                # train in batches
            val_loss, val_acc, val_base = self.validate()     # continious evaluation against unseen data

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.0%} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.0%} "
                  f"(Baselines training:{train_base:.0%} | testing: {val_base:.0%})")
            
            # Checkpoint: Only save model if it actually improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved! {save_path}")
                print(f"Val Loss: {val_loss:.4f}")