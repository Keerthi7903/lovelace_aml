import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import networkx as nx

# -------------------------------------------
# 1. IMPORT CUSTOM MODULES
# -------------------------------------------
try:
    from gnn_model import GNN  
    from transformer_model import TransactionTransformer 
    print("[SUCCESS] Loaded custom AI modules.")
except ImportError:
    print("[INFO] Custom models not found. Using internal fallback.")
    from torch_geometric.nn import SAGEConv
    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(2, 16)
            self.conv2 = SAGEConv(16, 1)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return torch.sigmoid(self.conv2(x, edge_index)).squeeze()
    
    class TransactionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)
        def forward(self, x):
            return torch.sigmoid(self.fc(x)).squeeze()

# -------------------------------------------
# 2. FEATURE ENGINEERING (With Time & Amount Logic)
# -------------------------------------------
def build_features_and_graph(df, node_map):
    N = len(node_map)
    # Features: 
    # 0: In-Degree
    # 1: Out-Degree
    # 2: Vol_In
    # 3: Vol_Out
    # 4: Peeling_Flag
    # 5: Cycle_Flag
    # 6: Token_Switch_Flag
    # 7: Burst_Flag (New: Rapid transactions)
    # 8: Flow_Match_Flag (New: In == Out)
    x_np = np.zeros((N, 9), dtype=np.float32)
    
    df = df.sort_values("Timestamp")
    
    # Tracking for Logic
    node_timestamps = defaultdict(list)
    tokens_in = defaultdict(set)
    tokens_out = defaultdict(set)
    last_amt = defaultdict(float)
    
    G_nx = nx.DiGraph()

    for _, row in df.iterrows():
        s = node_map.get(row['Source_Wallet_ID'])
        d = node_map.get(row['Dest_Wallet_ID'])
        if s is None or d is None: continue
        
        amt = float(row['Amount'])
        token = str(row['Token_Type']) 
        ts = row['Timestamp'].timestamp() # Convert to seconds
        
        G_nx.add_edge(s, d)
        
        # Basic Features
        x_np[s, 1] += 1 
        x_np[d, 0] += 1 
        x_np[s, 3] += amt
        x_np[d, 2] += amt
        
        # Time Tracking
        node_timestamps[s].append(ts)
        
        # Token Tracking
        tokens_out[s].add(token)
        tokens_in[d].add(token)
        
        # Peeling Logic (90-99%)
        if last_amt[s] > 0 and 0.90 < (amt / last_amt[s]) < 0.999:
            x_np[s, 4] = 1.0
        last_amt[s] = amt

    # --- ADVANCED LOGIC PROCESSING ---
    for i in range(N):
        # 1. Burst Detection (Rapid Timestamps)
        timestamps = sorted(node_timestamps[i])
        if len(timestamps) > 2:
            # If 3+ transactions happen in less than 60 seconds -> BURST
            duration = timestamps[-1] - timestamps[0]
            if duration < 60:
                x_np[i, 7] = 1.0 

        # 2. Flow Matching (In Amount ~= Out Amount)
        # If I received 1000 and sent 1000, I am likely a mule.
        vol_in = x_np[i, 2]
        vol_out = x_np[i, 3]
        if vol_in > 0 and vol_out > 0:
            ratio = vol_out / vol_in
            # If 98% to 102% match
            if 0.98 < ratio < 1.02:
                x_np[i, 8] = 1.0

        # 3. Token Switching
        t_in = tokens_in[i]
        t_out = tokens_out[i]
        if t_in and t_out and t_in.isdisjoint(t_out):
            x_np[i, 6] = 1.0 

    # Cycle Detection
    try:
        cycles = list(nx.simple_cycles(G_nx))
        for c in cycles:
            if len(c) > 2 and len(c) < 6:
                for n in c: x_np[n, 5] = 1.0
    except: pass
        
    return x_np, G_nx

# -------------------------------------------
# 3. SCENARIO HUNTER
# -------------------------------------------
def find_demo_scenarios(G_nx, risk_scores, node_map, x_np):
    scenarios = {}
    inv_map = {v: k for k, v in node_map.items()}
    risky_nodes = np.argsort(risk_scores)[::-1]
    
    for n in risky_nodes:
        # Scenario 1: Burst Smurfing (Fan-Out + Time)
        if x_np[n, 1] > 2 and x_np[n, 7] > 0:
            succ = list(G_nx.successors(n))
            if succ:
                scenarios["Scenario 1: Rapid Smurfing"] = {
                    "source": inv_map[n], "dest": inv_map[succ[0]],
                    "type": "Burst Fan-Out", "desc": "High volume sent in seconds."
                }
                break

    for n in risky_nodes:
        # Scenario 2: Aggregation
        if x_np[n, 0] > 2:
            preds = list(G_nx.predecessors(n))
            if preds:
                scenarios["Scenario 2: Aggregation"] = {
                    "source": inv_map[preds[0]], "dest": inv_map[n],
                    "type": "Fan-In Aggregation", "desc": "Funds aggregating at target."
                }
                break
    
    # Defaults if specialized ones not found
    if "Scenario 1: Rapid Smurfing" not in scenarios:
         for n in risky_nodes:
            if x_np[n, 1] > 2:
                scenarios["Scenario 1: Fan-Out"] = {"source": inv_map[n], "dest": inv_map[list(G_nx.successors(n))[0]], "type": "Fan-Out", "desc": "Funds split to mules."}; break

    return scenarios

# -------------------------------------------
# 4. MAIN EXECUTION
# -------------------------------------------
def main():
    print("[START] Initializing Maestro Engine...")
    
    if not os.path.exists('data/transactions.csv'):
        print("[ERROR] data/transactions.csv not found.")
        return

    df = pd.read_csv('data/transactions.csv', parse_dates=['Timestamp'])
    os.makedirs('prepared_data', exist_ok=True)
    
    # Load Labels
    known_illicit = set()
    if os.path.exists('data/labeled_wallets.csv'):
        try:
            lbl = pd.read_csv('data/labeled_wallets.csv')
            known_illicit = set(lbl['Wallet_ID'].astype(str).str.strip().values)
        except: pass

    # Map Nodes
    wallets = pd.concat([df['Source_Wallet_ID'], df['Dest_Wallet_ID']]).unique()
    node_map = {w: i for i, w in enumerate(wallets)}
    inv_map = {i: w for w, i in node_map.items()}
    
    # Extract Features
    print("[INFO] Extracting Graph Features...")
    x_np, G_nx = build_features_and_graph(df, node_map)

    # Prepare Tensors
    src_idx = df['Source_Wallet_ID'].map(node_map).values
    dst_idx = df['Dest_Wallet_ID'].map(node_map).values
    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
    
    # Run Models
    print("[INFO] Executing AI Models...")
    try:
        gnn_input = torch.tensor(x_np[:, :2]) # In/Out degree
        gnn_model = GNN()
        gnn_risk = gnn_model(gnn_input, edge_index).detach().numpy().flatten()
    except:
        gnn_risk = np.zeros(len(node_map))

    try:
        trans_input = torch.tensor(x_np[:, 4]).unsqueeze(1) # Peeling flag
        trans_model = TransactionTransformer()
        trans_risk = trans_model(trans_input).detach().numpy().flatten()
    except:
        trans_risk = np.zeros(len(node_map))

    # --- SCORING LOGIC (The "Flag" System) ---
    # Start with base score
    final_risk = (gnn_risk * 0.2) + (trans_risk * 0.2)
    
    # Apply "Flags" (The User's specific requirements)
    # 1. Burst Flag (Rapid timestamps) -> Huge Risk
    final_risk += (x_np[:, 7] * 0.4)
    
    # 2. Flow Match (In == Out) -> Mule Risk
    final_risk += (x_np[:, 8] * 0.3)
    
    # 3. Token Switch -> Suspicious
    final_risk += (x_np[:, 6] * 0.2)
    
    # 4. Topology (Fan Out)
    topo_score = np.log1p(x_np[:, 1]) / np.log1p(x_np[:, 1].max())
    final_risk += (topo_score * 0.3)

    # Seed Injection (Bad Guys are 1.0)
    seed_indices = [node_map[w] for w in known_illicit if w in node_map]
    for idx in seed_indices:
        final_risk[idx] = 1.0 
    
    # Propagate Risk to neighbors
    for seed_idx in seed_indices:
        for neighbor in G_nx.successors(seed_idx):
            final_risk[neighbor] = max(final_risk[neighbor], 0.85)
        for neighbor in G_nx.predecessors(seed_idx):
            final_risk[neighbor] = max(final_risk[neighbor], 0.75)

    final_risk = np.clip(final_risk, 0.05, 1.0) # Normal nodes get 0.05 (Green)

    # Reasoning Generation
    reasons = []
    for i in range(len(final_risk)):
        r = []
        if inv_map[i] in known_illicit: r.append("Known Illicit Wallet")
        if x_np[i, 7] > 0: r.append("Rapid Burst Activity (Time)")
        if x_np[i, 8] > 0: r.append("1:1 Flow Match (Mule)")
        if x_np[i, 1] > 3: r.append("High Fan-Out")
        if x_np[i, 0] > 3: r.append("High Fan-In")
        if x_np[i, 4] > 0: r.append("Peeling Chain")
        if x_np[i, 6] > 0: r.append("Token Switching")
        
        if not r: 
            if final_risk[i] > 0.5: r.append("Suspicious Topology")
            else: r.append("Normal Activity")
        reasons.append(" + ".join(r))

    # SAVE
    final_df = pd.DataFrame({
        "Wallet_ID": list(node_map.keys()),
        "Suspicion_Score": final_risk,
        "GNN_Score": gnn_risk,
        "Transformer_Score": trans_risk,
        "Reasoning": reasons
    })
    final_df.to_csv("prepared_data/final_risk_report.csv", index=False)
    
    scenarios = find_demo_scenarios(G_nx, final_risk, node_map, x_np)
    with open("prepared_data/demo_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)

    np.save("prepared_data/edge_index.npy", np.vstack([src_idx, dst_idx]))
    with open("prepared_data/node_map.json", "w") as f:
        json.dump(node_map, f)
        
    print(f"[DONE] Analysis Complete!")

if __name__ == '__main__':
    main()