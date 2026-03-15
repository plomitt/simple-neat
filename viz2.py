import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(net):
    G = nx.DiGraph()
    color_map =[]
    
    # 1. Add Nodes
    for node in net.all_nodes:
        # Determine color
        if node in net.inputs:
            color = 'lightblue'
        elif node in net.outputs:
            color = 'lightgreen'
        else:
            color = 'lightcoral'
            
        color_map.append(color)
        
        # Add node with metadata for the label
        G.add_node(
            node.label, 
            activation=node.activation_cls.__name__, 
            bias=round(node.bias.data, 2)
        )
        
        # 2. Add Edges (Connections)
        for conn in node.connections_out:
            G.add_edge(
                conn.source.label, 
                conn.target.label, 
                weight=round(conn.weight.data, 2)
            )
            
    # 3. Draw the Graph
    pos = nx.spring_layout(G, seed=42) # Spring layout spaces things out nicely
    
    # Create text labels for nodes (Name + Activation + Bias)
    labels = {n: f"{n}\n{G.nodes[n]['activation']}\nb: {G.nodes[n]['bias']}" for n in G.nodes}
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=color_map, with_labels=True, labels=labels, 
            node_size=3000, font_size=9, font_weight='bold', arrows=True)
    
    # Draw weight labels on the edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Evolved Neural Network Topology")
    plt.show()