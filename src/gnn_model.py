import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(2, 16)
        self.conv2 = SAGEConv(16, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x).squeeze()
