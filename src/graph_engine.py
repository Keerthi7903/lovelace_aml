import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphEngine(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Using GraphSAGE to handle large-scale blockchain graphs
        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 1)

    def forward(self, x, edge_index):
        # x: node features, edge_index: connections
        x = F.relu(self.conv1(x, edge_index))
        return torch.sigmoid(self.conv2(x, edge_index)) # Risk Probability