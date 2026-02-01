import torch
import torch.nn as nn

class TransactionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(1, 16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embed(x)      # [N, 16]
        x = x.unsqueeze(1)     # [N, 1, 16]
        x = self.encoder(x)    # [N, 1, 16]
        x = x.squeeze(1)       # [N, 16]
        return torch.sigmoid(self.fc(x)).squeeze()
