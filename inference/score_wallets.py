import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from torch_geometric.utils import from_networkx

from graph.build_graph import build_transaction_graph
from graph.feature_engineering import extract_features
from models.gnn_model import GNN

df = pd.read_csv("data/transactions.csv")
G = build_transaction_graph(df)

x, nodes = extract_features(G)
data = from_networkx(G)
data.x = x

model = GNN()
model.load_state_dict(torch.load("models/gnn_model.pt"))
model.eval()

with torch.no_grad():
    scores = model(data.x, data.edge_index)

result = pd.DataFrame({
    "Wallet": nodes,
    "Risk_Score": scores.numpy()
}).sort_values("Risk_Score", ascending=False)

print(result.head(10))
