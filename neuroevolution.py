import random
import copy
from nn_graph import Network, Neuron, Connection, Sigmoid, ReLU, Tanh, Linear

# ==========================================
# 1. Innovation Tracker
# ==========================================
class Innovator:
    """Tracks global historical markers to align genomes during crossover (mating)."""
    def __init__(self):
        self.current_innov = 0
        self.connection_history = {} # (source_id, target_id) -> innov_number
        self.current_node_id = 0
        
    def get_innov(self, source, target):
        if (source, target) not in self.connection_history:
            self.current_innov += 1
            self.connection_history[(source, target)] = self.current_innov
        return self.connection_history[(source, target)]
        
    def get_node_id(self):
        self.current_node_id += 1
        return self.current_node_id

# ==========================================
# 2. Genes
# ==========================================
class NodeGene:
    def __init__(self, node_id, node_type, bias=random.uniform(-1.0, 1.0), activation=Sigmoid):
        self.id = node_id
        self.type = node_type # 'input', 'hidden', 'output'
        self.bias = bias
        self.activation = activation

class ConnGene:
    def __init__(self, innov, source, target, weight):
        self.innov = innov
        self.source = source
        self.target = target
        self.weight = weight

# ==========================================
# 3. Genome
# ==========================================
class Genome:
    def __init__(self):
        self.nodes = {} # id -> NodeGene
        self.conns = {} # innov -> ConnGene
        self.fitness = 0.0
        self.is_innovative = False # Flag for 1-generation immunity

        self.fitness_history = []
        self.effective_fitness = 0.0

    def build_phenotype(self):
        """Translates the Genome data into a physical, runnable Network graph."""
        net = Network()
        physical_nodes = {}
        
        # 1. Build physical neurons
        for node_id, ng in self.nodes.items():
            neuron = Neuron(ng.activation, label=f"{ng.type}_{node_id}")
            neuron.bias.data = ng.bias
            physical_nodes[node_id] = neuron
            net.add_node(neuron, ng.type)
            
        # 2. Wire physical connections
        for cg in self.conns.values():
            if cg.source in physical_nodes and cg.target in physical_nodes:
                src, tgt = physical_nodes[cg.source], physical_nodes[cg.target]
                net.add_connection(src, tgt, weight_val=cg.weight)
                
        return net

    def crossover(self, other, config):
        child = Genome()
        # Ensure we always treat 'self' as the fitter one
        
        # 1. Inherit Nodes: Always take from fitter, but keep existing structure
        for nid, n in self.nodes.items():
            child.nodes[nid] = NodeGene(n.id, n.type, n.bias, n.activation)
            
        # 2. Inherit Connections
        # Combine keys from both parents
        all_innovs = set(self.conns.keys()) | set(other.conns.keys())
        
        for innov in all_innovs:
            if innov in self.conns and innov in other.conns:
                # Matching genes: 50/50 weight inheritance
                c = random.choice([self.conns[innov], other.conns[innov]])
                child.conns[innov] = ConnGene(innov, c.source, c.target, c.weight)
            elif innov in self.conns:
                # Disjoint/Excess from fitter: always keep
                c = self.conns[innov]
                child.conns[innov] = ConnGene(innov, c.source, c.target, c.weight)
            else:
                # Disjoint/Excess from less-fit: keep with 20% probability (Diversity injection)
                if random.random() < config["prob_diversity_injection"]:
                    c = other.conns[innov]
                    child.conns[innov] = ConnGene(innov, c.source, c.target, c.weight)
                    
        return child

    def mutate(self, innovator, config, available_activations=[Sigmoid, ReLU, Tanh]):
        mut_logs = []
        hiddens = [n for n in self.nodes.values() if n.type == 'hidden']
        non_inputs = [n for n in self.nodes.values() if n.type != 'input']

        # 1. Parameter Mutations
        if random.random() < config["prob_mutate_weight"]:
            mut_logs.append("W_Mut")
            for c in self.conns.values():
                r = random.random()
                if r < 0.4: c.weight += random.uniform(-0.5, 0.5) 
                elif r < 0.8: c.weight += random.uniform(-0.05, 0.05) 
                else: c.weight = random.uniform(-1.0, 1.0) 

        if random.random() < config["prob_mutate_bias"]:
            mut_logs.append("B_Mut")
            for n in non_inputs:
                if random.random() < 0.5: n.bias += random.uniform(-0.5, 0.5)
                    
        if random.random() < config["prob_mutate_act"] and hiddens:
            mut_logs.append("Act_Mut")
            random.choice(hiddens).activation = random.choice(available_activations)

        # 2. Structural Mutations
        struct_roll = random.random()
        
        if struct_roll < config["prob_add_node"] and self.conns:
            innov = random.choice(list(self.conns.keys()))
            old_c = self.conns.pop(innov)
            new_id = innovator.get_node_id()
            self.nodes[new_id] = NodeGene(new_id, 'hidden', random.uniform(-1.0, 1.0), Sigmoid)
            self.is_innovative = True
            
            in1 = innovator.get_innov(old_c.source, new_id)
            in2 = innovator.get_innov(new_id, old_c.target)
            self.conns[in1] = ConnGene(in1, old_c.source, new_id, 1.0)
            self.conns[in2] = ConnGene(in2, new_id, old_c.target, old_c.weight)
            mut_logs.append("+Node")
            return mut_logs
            
        if struct_roll < config["prob_add_conn"]:
            n1 = random.choice(list(self.nodes.values())) 
            n2 = random.choice(non_inputs)                
            if not any(c.source == n1.id and c.target == n2.id for c in self.conns.values()):
                if (n1.id, n2.id) not in innovator.connection_history:
                    self.is_innovative = True
                innov = innovator.get_innov(n1.id, n2.id)
                self.conns[innov] = ConnGene(innov, n1.id, n2.id, random.uniform(-1.0, 1.0))
                mut_logs.append("+Conn")
            return mut_logs
            
        if struct_roll < config["prob_rm_node"] and hiddens:
            n = random.choice(hiddens)
            del self.nodes[n.id]
            orphans =[i for i, c in self.conns.items() if c.source == n.id or c.target == n.id]
            for i in orphans: del self.conns[i]
            mut_logs.append("-Node")
            return mut_logs
            
        if struct_roll < config["prob_rm_conn"] and self.conns:
            innov = random.choice(list(self.conns.keys()))
            del self.conns[innov]
            mut_logs.append("-Conn")
            return mut_logs
            
        if struct_roll < config["prob_swap"] and len(hiddens) >= 2:
            n1, n2 = random.sample(hiddens, 2)
            n1.activation, n2.activation = n2.activation, n1.activation
            n1.bias, n2.bias = n2.bias, n1.bias
            mut_logs.append("Swap")
            return mut_logs

        return mut_logs

# ==========================================
# 4. Evolution Manager
# ==========================================
class EvolutionManager:
    def __init__(self, input_size, output_size, pop_size=100):
        self.innovator = Innovator()
        self.pop_size = pop_size
        self.population = []
        
        # Pre-allocate static input/output IDs so all genomes share the same I/O anchors
        self.base_inputs = [self.innovator.get_node_id() for _ in range(input_size)]
        self.base_outputs = [self.innovator.get_node_id() for _ in range(output_size)]

        # Initialize Population with minimal networks (Inputs directly wired to Outputs)
        for _ in range(pop_size):
            g = Genome()
            for i in self.base_inputs:
                g.nodes[i] = NodeGene(i, 'input', 0.0, Linear)
            for o in self.base_outputs:
                g.nodes[o] = NodeGene(o, 'output', random.uniform(-1.0, 1.0), Sigmoid)
                
            for i in self.base_inputs:
                for o in self.base_outputs:
                    innov = self.innovator.get_innov(i, o)
                    g.conns[innov] = ConnGene(innov, i, o, random.uniform(-1.0, 1.0))
            self.population.append(g)

    def evolve(self, fitness_fn, generations=100, fitness_threshold=None):
        for gen in range(generations):
            # 1. Evaluate
            for g in self.population:
                net = g.build_phenotype()
                g.fitness = fitness_fn(net)
                
            # 2. Sort (Highest fitness first)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best_score = self.population[0].fitness
            print(f"Gen {gen+1:03d} | Best Fitness: {best_score:.4f}")

            # Early Stop
            if fitness_threshold is not None and best_score >= fitness_threshold:
                print(f"\nSolved! Target fitness {fitness_threshold} reached in Generation {gen+1}.")
                break
            
            # 3. Next Generation Elitism & Innovation Protection
            next_gen = []
            elites_count = max(1, int(self.pop_size * 0.1)) # Keep top 10%
            next_gen.extend(self.population[:elites_count])
            
            # Protect Innovators
            for g in self.population[elites_count:]:
                if g.is_innovative and len(next_gen) < self.pop_size:
                    next_gen.append(g)
            
            # Reset immunity for all survivors
            for g in next_gen:
                g.is_innovative = False
            
            # 4. Breed & Mutate to fill the rest
            while len(next_gen) < self.pop_size:
                p1 = random.choice(self.population[:self.pop_size // 2])
                p2 = random.choice(self.population[:self.pop_size // 2])
                
                if p2.fitness > p1.fitness:
                    p1, p2 = p2, p1
                    
                child = p1.crossover(p2)
                
                if random.random() < 0.80:
                    child.mutate(self.innovator)
                    
                next_gen.append(child)
                
            self.population = next_gen
            
        return self.population[0].build_phenotype()