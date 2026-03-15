import networkx as nx
import matplotlib.pyplot as plt
import json

def visualize_network(source):
    """
    Accepts either a Network object OR a path to a JSON file.
    Uses a hierarchical multipartite layout for readability.
    """
    G = nx.DiGraph()
    layers = {"input": [], "hidden": [], "output": []}
    
    # --- PARSING ---
    if isinstance(source, str):  # If JSON file path
        with open(source, 'r') as f: data = json.load(f)
        for n in data["nodes"]:
            G.add_node(n["label"], activation=n["activation"], bias=round(n["bias"], 2), type=n["type"])
            layers[n["type"]].append(n["label"])
        for c in data["connections"]:
            src = next(n["label"] for n in data["nodes"] if n["id"] == c["source_id"])
            tgt = next(n["label"] for n in data["nodes"] if n["id"] == c["target_id"])
            G.add_edge(src, tgt, weight=round(c["weight"], 2))
    else:  # If Network object
        for node in source.all_nodes:
            ntype = "input" if node in source.inputs else ("output" if node in source.outputs else "hidden")
            G.add_node(node.label, activation=node.activation_cls.__name__, bias=round(node.bias.data, 2), type=ntype)
            layers[ntype].append(node.label)
        for node in source.all_nodes:
            for conn in node.connections_out:
                G.add_edge(conn.source.label, conn.target.label, weight=round(conn.weight.data, 2))

    # --- LAYOUT (Hierarchical) ---
    pos = {}
    # Spread nodes vertically in columns
    for i, n in enumerate(layers["input"]): pos[n] = (0, i)
    for i, n in enumerate(layers["hidden"]): pos[n] = (1, i)
    for i, n in enumerate(layers["output"]): pos[n] = (2, i)

    # --- DRAWING ---
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color='skyblue', arrows=True)
    
    # Labels with Activation/Bias
    labels = {n: f"{n}\n{G.nodes[n].get('activation','')}\nb: {G.nodes[n].get('bias','')}" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Weights on edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Evolved Network Hierarchy")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_network("best_net/xor_champion.json")