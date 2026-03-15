import json

def save_network_to_json(net, filename="best_network.json"):
    """Saves the network topology, weights, and biases to a JSON file."""
    
    # Map physical objects to temporary unique IDs for serialization
    node_to_id = {node: i for i, node in enumerate(net.all_nodes)}
    
    data = {
        "nodes":[],
        "connections":[]
    }
    
    # 1. Save Nodes
    for node, n_id in node_to_id.items():
        node_type = "hidden"
        if node in net.inputs: node_type = "input"
        elif node in net.outputs: node_type = "output"
        
        data["nodes"].append({
            "id": n_id,
            "label": node.label,
            "type": node_type,
            "bias": node.bias.data,
            "activation": node.activation_cls.__name__
        })
        
        # 2. Save Connections originating from this node
        for conn in node.connections_out:
            data["connections"].append({
                "source_id": node_to_id[conn.source],
                "target_id": node_to_id[conn.target],
                "weight": conn.weight.data,
                "gater_id": node_to_id[conn.gater] if conn.gater else None
            })
            
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Network saved successfully to {filename}")