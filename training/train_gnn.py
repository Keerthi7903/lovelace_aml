import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from torch_geometric.utils import from_networkx

from graph.build_graph import build_transaction_graph
from graph.feature_engineering import extract_features
from models.gnn_model import GNN

# CONFIG
CSV_PATH = "data/transactions.csv"
EPOCHS = 10

# LOAD DATA
df = pd.read_csv(CSV_PATH)
G = build_transaction_graph(df)

x, nodes = extract_features(G)
# Load illicit wallet seeds
illicit_df = pd.read_csv("data/illicit_wallets.csv")

# Create set of known illicit wallets
illicit_wallets = set(
    illicit_df[illicit_df["Label"] == "illicit"]["Wallet_ID"]
)

# Supervised labels: 1 = illicit, 0 = normal
labels = torch.tensor(
    [1 if node in illicit_wallets else 0 for node in nodes],
    dtype=torch.float
)

data = from_networkx(G)
data.x = x

# MODEL
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

print("🚀 Training GNN")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {loss.item():.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gnn_model.pt")
print("✅ Model saved")
