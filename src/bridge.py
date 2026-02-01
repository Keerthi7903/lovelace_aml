import pandas as pd
import numpy as np
import torch

def finalize_risk_report(node_map, gnn_probs, behavioral_sigs):
    """
    Combines spatial (GNN) and temporal (Transformer) insights.
    """
    report = []
    inv_map = {v: k for k, v in node_map.items()}
    
    for i, wallet_id in inv_map.items():
        # Weighted Scoring Logic
        risk_val = (0.7 * float(gnn_probs[i])) + (0.3 * float(behavioral_sigs[i].norm() / 10))
        
        report.append({
            "Wallet_ID": wallet_id,
            "Risk_Score": round(risk_val, 4),
            "Status": "HIGH RISK" if risk_val > 0.75 else "NORMAL"
        })
    
    df = pd.DataFrame(report)
    df.to_csv("prepared_data/final_risk_report.csv", index=False)
    print("✅ Intelligence Report Generated!")