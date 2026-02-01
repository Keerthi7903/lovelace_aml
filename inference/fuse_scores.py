import torch
import pandas as pd
from models.gnn_model import GNN
from models.transformer_model import TransactionTransformer
from graph.build_graph import build_transaction_graph

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/transactions.csv")

# -----------------------------
# LOAD GRAPH
# -----------------------------
G, node_map = build_transaction_graph(df)

# Node features: in-degree & out-degree
x = torch.tensor(
    [[G.out_degree(n), G.in_degree(n)] for n in G.nodes()],
    dtype=torch.float
)

edge_index = torch.tensor(
    [
        [node_map[src], node_map[dst]]
        for src, dst in G.edges()
    ],
    dtype=torch.long
).t().contiguous()


# -----------------------------
# LOAD GNN
# -----------------------------
gnn = GNN()
gnn.load_state_dict(torch.load("models/gnn_model.pt"))
gnn.eval()

with torch.no_grad():
    gnn_scores = gnn(x, edge_index)

wallet_scores = dict(zip(G.nodes(), gnn_scores.numpy()))

# -----------------------------
# LOAD TRANSFORMER (NO TRAINING)
# -----------------------------
amounts = torch.tensor(
    df["Amount"].values,
    dtype=torch.float
).unsqueeze(1)

transformer = TransactionTransformer()
transformer.load_state_dict(torch.load("models/transformer_model.pt"))
transformer.eval()

with torch.no_grad():
    transformer_scores = transformer(amounts).numpy()

df["Transformer_Risk"] = transformer_scores
df["GNN_Risk"] = df["Source_Wallet_ID"].map(wallet_scores).fillna(0)

# -----------------------------
# FUSION
# -----------------------------
df["Final_Risk"] = 0.6 * df["GNN_Risk"] + 0.4 * df["Transformer_Risk"]

threshold = df["Final_Risk"].quantile(0.95)
df["Label"] = df["Final_Risk"].apply(
    lambda x: "illicit" if x >= threshold else "legal"
)

# -----------------------------
# OUTPUT
# -----------------------------
print(df[[
    "Source_Wallet_ID",
    "Dest_Wallet_ID",
    "Final_Risk",
    "Label"
]].head(10))

print("\n🚨 Top Suspicious Wallets")
print(
    df.groupby("Source_Wallet_ID")["Final_Risk"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)
