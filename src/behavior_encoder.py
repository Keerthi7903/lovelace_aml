import torch
import torch.nn as nn

class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=32, nhead=4):
        super().__init__()
        # Input features: [direction, amt_log1p, time_rel, peel_indicator]
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), 
            num_layers=2
        )
        self.fc = nn.Linear(d_model, 16)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = self.embedding(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)) # Behavioral Embedding