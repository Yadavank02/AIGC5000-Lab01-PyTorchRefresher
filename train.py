import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

opt_choice = sys.argv[1].lower() if len(sys.argv) > 1 else 'adam'
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" )

#Cleaning and Processing Data
df = pd.read_csv("real_estate.csv")

print("Missing values per column")
print(df.isnull().sum())

plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(), cbar = False, cmap='viridis', yticklabels = False)
plt.title("Missing Data")
plt.xlabel("Columns")
plt.tight_layout()
plt.savefig("Missing_values_heatmap.png")
plt.close()
print("Missing Values file saved")

feature_cols = [col for col in df.columns if col!= "price_per_unit"]

pairplot = sns.pairplot(df, y_vars= ["price_per_unit"], x_vars=feature_cols, height=2.5)
pairplot.savefig("Features vs Price.png")
plt.close()

df_clean = df.dropna().copy()

#Data Normalization
features = ["house_age", "transit_distance", "local_convenience_stores", "latitude", "longitude"]
X_raw = df_clean[features].values.astype('float32')
y_raw = df_clean['price_per_unit'].values.astype('float32').reshape(-1, 1)
X_scaled = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

# Custom Dataset & DataLoader 
class HouseDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features)
        self.y = torch.tensor(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = HouseDataset(X_scaled, y_raw)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class HouseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First Hidden Layer: 5 inputs -> 32 neurons
        self.hidden1 = nn.Linear(5, 32)
        self.relu1 = nn.ReLU()
        
        # Second Hidden Layer: 32 neurons -> 16 neurons
        self.hidden2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        
        # Output Layer: 16 neurons -> 1 prediction output
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

model = HouseNet().to(device)
criterion = nn.MSELoss()
if opt_choice == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"\n Starting Training (Optimizer: {opt_choice.upper()} | LR: {learning_rate} | Device: {device})")
for epoch in range(1, 101):
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()                   
        predictions = model(x_batch)            
        loss = criterion(predictions, y_batch) 
        loss.backward()                         
        optimizer.step()                       
        
        total_loss += loss.item() * len(x_batch)

    average_loss = total_loss / len(dataset)
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/100 | Loss: {average_loss:.4f}")

print(f"Final Loss for Optimizer={opt_choice.upper()} (lr={learning_rate}: {average_loss:.4f}")

log_line = f"Seed: {SEED} | Optimizer: {opt_choice.upper()} | LR: {learning_rate} | Layers: 2 Hidden (32, 16) | Final Loss: {average_loss:.4f}\n"
with open("experiment_log.txt", "a") as f:
    f.write(log_line)
print("Log entry saved to 'experiment_log.txt'.\n")