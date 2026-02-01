import torch
import pandas as pd
import torch.nn as nn

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/transactions.csv")

# Feature: transaction amount (normalized)
amounts = torch.tensor(
    df["Amount"].values,
    dtype=torch.float
).unsqueeze(1)  # [N, 1]

# Weak supervision label (hackathon-safe heuristic)
# Small repeated amounts → suspicious
threshold = df["Amount"].quantile(0.2)
labels = torch.tensor(
    [1 if a < threshold else 0 for a in df["Amount"]],
    dtype=torch.float
)

# -----------------------------
# TRANSFORMER MODEL
# -----------------------------
class TxnTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embed(x)          # [N, 16]
        x = x.unsqueeze(1)         # [N, 1, 16]
        x = self.encoder(x)        # [N, 1, 16]
        x = x.squeeze(1)           # [N, 16]
        return torch.sigmoid(self.fc(x)).squeeze()  # [N]

model = TxnTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# -----------------------------
# TRAINING
# -----------------------------
print("🚀 Training Transformer")

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(amounts)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/5 | Loss {loss.item():.4f}")

torch.save(model.state_dict(), "models/transformer_model.pt")
print("✅ Transformer model saved")
