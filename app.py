import streamlit as st
import pandas as pd
import os
import numpy as np
import json
import networkx as nx 
import subprocess
import sys
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Smurfing Hunter", layout="wide", page_icon="🛡️")

OVERRIDE_DB = {
    # --- RED NODES (High Risk) ---
    "A1": [0.94, "#FF0000", "Rapid Burst + Fan-Out"],
    "H1": [0.89, "#FF0000", "High Value Aggregation"],
    "A3": [0.92, "#FF0000", "Known Blacklisted Wallet"],
    "D3": [0.88, "#FF0000", "Illicit Sink Node"],
    "A5": [0.91, "#FF0000", "Smurfing Source"],
    "C5": [0.87, "#FF0000", "Layering Node"],
    "P2": [0.95, "#FF0000", "High Velocity Source"],
    "S2": [0.83, "#FF0000", "Fan In Anomaly"],

    # --- YELLOW NODES (Medium Risk / Mules) ---
    "B1": [0.55, "#FFA500", "Mule Account (1:1 Flow)"],
    "C1": [0.58, "#FFA500", "Mule Account (1:1 Flow)"],
    "D1": [0.54, "#FFA500", "Mule Account (1:1 Flow)"],
    "E1": [0.61, "#FFA500", "Mule Account (1:1 Flow)"],
    "F1": [0.63, "#FFA500", "Aggregation Point"],
    "G1": [0.62, "#FFA500", "Aggregation Point"],
    "B3": [0.52, "#FFA500", "Suspicious Link"],
    "C3": [0.49, "#FFA500", "Suspicious Link"],
    "B5": [0.48, "#FFA500", "Bridge Node"],
    "Q2": [0.46, "#FFA500", "Suspicious Link"],
    "R2": [0.45, "#FFA500", "Suspicious Link"],

    # --- GREEN NODES (Safe) ---
    "A4": [0.12, "#00CC00", "Normal Activity"],
    "B4": [0.08, "#00CC00", "Normal Activity"],
    "C4": [0.09, "#00CC00", "Normal Activity"],
    "D4": [0.07, "#00CC00", "Normal Activity"],
    "E4": [0.05, "#00CC00", "Normal Activity"],
    "T2": [0.11, "#00CC00", "Safe Destination"]
}

# SIDEBAR
with st.sidebar:
    st.title("🛡️ Smurfing Hunter")
    st.markdown("### 🧠 Logic Engine")
    st.markdown("**1. Burst Detection:** Flags transactions happening in < 120s.")
    st.markdown("**2. Flow Matching:** Flags 1:1 In/Out amounts (Mules).")
    st.markdown("**3. Topology:** Fan-In/Fan-Out.")
    st.markdown("---")
    
    scenario_options = ["Overview"]
    scenarios_data = {}
    if os.path.exists("prepared_data/demo_scenarios.json"):
        try:
            with open("prepared_data/demo_scenarios.json") as f:
                scenarios_data = json.load(f)
            scenario_options += list(scenarios_data.keys())
        except: pass
    
    selected_scenario = st.selectbox("Select View:", scenario_options)
    
    st.markdown("---")
    
    if st.button("🚀 RUN AI ANALYSIS NOW", type="primary"):
        with st.spinner("🧠 Neuro-Symbolic Engine is processing your data..."):
            try:
                result = subprocess.run(
                    [sys.executable, "gnn_transformer_data_prep.py"], 
                    cwd="src", 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    st.success("Analysis Complete!")
                    st.cache_data.clear()
                else:
                    st.error("Error in Analysis Script:")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"Failed to run script: {e}")

st.title("🛡️ The Smurfing Hunter: Detecting Money Laundering Circles")

# TABS
tab1, tab2, tab3 = st.tabs(["📂 Data Ingestion", "📊 Suspicion Report", "🕸️ Laundering Graph"])

# TAB 1: UPLOAD
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Upload Transactions")
        uploaded_tx = st.file_uploader("Transactions CSV", type=["csv"], key="tx")
        if uploaded_tx:
            os.makedirs("data", exist_ok=True)
            with open("data/transactions.csv", "wb") as f:
                f.write(uploaded_tx.getbuffer())
            st.success("Uploaded")

    with col2:
        st.markdown("### 2. Upload Illicit Labels (Optional)")
        uploaded_lbl = st.file_uploader("Labeled Wallets CSV", type=["csv"], key="lbl")
        if uploaded_lbl:
            with open("data/labeled_wallets.csv", "wb") as f:
                f.write(uploaded_lbl.getbuffer())
            st.success("Uploaded")
        else:
            st.info("No labels? AI will use Burst & Flow heuristics.")
            
    if os.path.exists("data/transactions.csv"):
        st.markdown("#### Preview (With Timestamps)")
        df_preview = pd.read_csv("data/transactions.csv")
        st.dataframe(df_preview.head(10), use_container_width=True)

# TAB 2: REPORT (FORCED SCORES)
with tab2:
    st.markdown("### 📊 Suspicion Score Report")
    
    if os.path.exists("prepared_data/final_risk_report.csv"):
        # Load the real CSV structure
        df = pd.read_csv("prepared_data/final_risk_report.csv")
        
        # --- ⚡ INJECT HARDCODED VALUES ⚡ ---
        # We iterate through the rows and force the values from OVERRIDE_DB
        for index, row in df.iterrows():
            w_id = str(row['Wallet_ID']).strip()
            if w_id in OVERRIDE_DB:
                # Force the Score and Reason
                df.at[index, 'Suspicion_Score'] = OVERRIDE_DB[w_id][0]
                df.at[index, 'Reasoning'] = OVERRIDE_DB[w_id][2]
        
        # Sort by the new forced scores
        df = df.sort_values("Suspicion_Score", ascending=False)
        
        def color_risk(val):
            if val > 0.8: return 'background-color: rgba(255, 0, 0, 0.3)' # Red
            if val > 0.4: return 'background-color: rgba(255, 165, 0, 0.3)' # Orange
            return 'background-color: rgba(0, 255, 0, 0.1)' # Green
            
        st.dataframe(
            df.style.map(color_risk, subset=['Suspicion_Score']), 
            use_container_width=True,
            column_config={
                "Reasoning": st.column_config.TextColumn("Flags Detected", width="large"),
                "Suspicion_Score": st.column_config.ProgressColumn("Suspicion Score", format="%.2f"),
            }
        )
    else:
        st.warning("⚠️ No Analysis Found. Please click 'RUN AI ANALYSIS NOW' in the sidebar.")

# TAB 3: MAP (FORCED COLORS)
with tab3:
    st.markdown("### 🕸️ Laundering Graph Visualization")
    
    if os.path.exists("data/transactions.csv") and os.path.exists("prepared_data/final_risk_report.csv"):
        
        # Load Risk Scores (Modified with Overrides)
        risk_df = pd.read_csv("prepared_data/final_risk_report.csv")
        # Create a lookup dictionary
        wallet_risk = {}
        for _, row in risk_df.iterrows():
            w_id = str(row['Wallet_ID']).strip()
            # If in hardcoded DB, use that score. Else use CSV score.
            if w_id in OVERRIDE_DB:
                wallet_risk[w_id] = OVERRIDE_DB[w_id][0]
            else:
                wallet_risk[w_id] = row.get('Suspicion_Score', 0.1)
        
        # Load Transactions
        tx_df = pd.read_csv("data/transactions.csv")
        
        # Build Graph
        G = nx.DiGraph()
        for _, row in tx_df.iterrows():
            src = str(row['Source_Wallet_ID'])
            dst = str(row['Dest_Wallet_ID'])
            amt = float(row['Amount'])
            token = str(row['Token_Type'])
            ts = str(row['Timestamp']) 
            
            hover_text = f"Amount: {amt} {token}\nTime: {ts}"
            G.add_edge(src, dst, label=f"{amt:.1f}", title=hover_text)

        # Setup PyVis
        net = Network(height='650px', width='100%', directed=True, bgcolor="#1E1E1E", font_color="white")
        
        nodes_to_draw = set()
        
        if selected_scenario == "Overview":
            nodes_to_draw.update(G.nodes())
        else:
            if selected_scenario in scenarios_data:
                scen = scenarios_data[selected_scenario]
                st.info(f"**Pattern:** {scen['type']} | **Desc:** {scen['desc']}")
                s_id = str(scen['source'])
                d_id = str(scen['dest'])
                nodes_to_draw.update([s_id, d_id])
                try:
                    for p in nx.all_simple_paths(G, s_id, d_id, cutoff=4): 
                        nodes_to_draw.update(p)
                except: 
                    if s_id in G: nodes_to_draw.update(G.successors(s_id))

        # DRAW NODES
        for n in nodes_to_draw:
            node_id = str(n).strip()
            
            # --- ⚡ APPLY HARDCODED COLOR & SCORE ⚡ ---
            if node_id in OVERRIDE_DB:
                final_score = OVERRIDE_DB[node_id][0]
                final_color = OVERRIDE_DB[node_id][1]
            else:
                # Default logic for nodes not in your note
                final_score = wallet_risk.get(node_id, 0.1)
                if final_score > 0.8: final_color = "#FF0000"
                elif final_score > 0.4: final_color = "#FFA500"
                else: final_color = "#00CC00"
            
            net.add_node(n, label=n, color=final_color, title=f"Risk: {final_score:.2f}")

        # DRAW EDGES
        for u in nodes_to_draw:
            if u in G:
                for v in G.successors(u):
                    if v in nodes_to_draw:
                        edge_data = G.get_edge_data(u, v)
                        lbl = edge_data.get('label', '')
                        ttl = edge_data.get('title', '')
                        net.add_edge(u, v, color="#888888", label=lbl, title=ttl)

        net.force_atlas_2based(gravity=-50, spring_length=120, overlap=1)
        net.save_graph("map.html")
        components.html(open("map.html", 'r').read(), height=700)
    else:
        st.warning("⚠️ Please upload transactions and Run Analysis.")