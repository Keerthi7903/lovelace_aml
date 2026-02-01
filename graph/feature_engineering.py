import torch

def extract_features(G):
    features = []
    nodes = list(G.nodes())
    for n in nodes:
        features.append([G.out_degree(n), G.in_degree(n)])
    return torch.tensor(features, dtype=torch.float), nodes
