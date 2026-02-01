import networkx as nx

def build_transaction_graph(df):
    G = nx.DiGraph()
    node_map = {}

    for _, row in df.iterrows():
        src = row["Source_Wallet_ID"]
        dst = row["Dest_Wallet_ID"]

        G.add_edge(src, dst, amount=row["Amount"])

    node_map = {node: idx for idx, node in enumerate(G.nodes())}

    return G, node_map
