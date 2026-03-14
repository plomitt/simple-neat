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
    def __init__(self, node_id, node_type, bias=0.0, activation=Sigmoid):
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

    def crossover(self, other):
        """Mates two genomes. Assumes `self` is the fitter parent."""
        child = Genome()
        
        # Inherit all topology (nodes) from the fitter parent
        for node_id, n in self.nodes.items():
            child.nodes[node_id] = NodeGene(n.id, n.type, n.bias, n.activation)
            
        # Inherit connections
        for innov, c in self.conns.items():
            # If both parents have the connection, 50/50 chance for the weight
            if innov in other.conns:
                weight = random.choice([c.weight, other.conns[innov].weight])
            else:
                # Disjoint/Excess genes: Inherit from the fitter parent (self)
                weight = c.weight
            child.conns[innov] = ConnGene(innov, c.source, c.target, weight)
            
        return child

    def mutate(self, innovator, available_activations=[Sigmoid, ReLU, Tanh]):
        """Selects and applies a single random mutation."""
        mutations = ['act', 'bias', 'swap', 'add_conn', 'rm_conn', 'weight', 'add_node', 'rm_node']
        choice = random.choice(mutations)
        
        # Helper lists
        hiddens = [n for n in self.nodes.values() if n.type == 'hidden']
        modifiables = [n for n in self.nodes.values() if n.type != 'input']

        if choice == 'act' and modifiables:
            n = random.choice(modifiables)
            n.activation = random.choice(available_activations)
            
        elif choice == 'bias' and modifiables:
            n = random.choice(modifiables)
            n.bias += random.uniform(-0.5, 0.5)
            
        elif choice == 'swap' and len(hiddens) >= 2:
            n1, n2 = random.sample(hiddens, 2)
            n1.activation, n2.activation = n2.activation, n1.activation
            n1.bias, n2.bias = n2.bias, n1.bias
            
        elif choice == 'add_conn':
            n1 = random.choice(list(self.nodes.values()))
            n2 = random.choice(modifiables) # Don't target inputs
            if not any(c.source == n1.id and c.target == n2.id for c in self.conns.values()):
                innov = innovator.get_innov(n1.id, n2.id)
                self.conns[innov] = ConnGene(innov, n1.id, n2.id, random.uniform(-1, 1))
                
        elif choice == 'rm_conn' and self.conns:
            innov = random.choice(list(self.conns.keys()))
            del self.conns[innov]
            
        elif choice == 'weight' and self.conns:
            c = random.choice(list(self.conns.values()))
            c.weight += random.uniform(-0.5, 0.5)
            
        elif choice == 'add_node' and self.conns:
            # 1. Pick and remove a connection
            innov = random.choice(list(self.conns.keys()))
            old_c = self.conns.pop(innov)
            
            # 2. Add the new node
            new_id = innovator.get_node_id()
            self.nodes[new_id] = NodeGene(new_id, 'hidden', 0.0, Sigmoid)
            
            # 3. Wire it in: Source -> New (weight 1) -> Target (old weight)
            in1 = innovator.get_innov(old_c.source, new_id)
            in2 = innovator.get_innov(new_id, old_c.target)
            
            self.conns[in1] = ConnGene(in1, old_c.source, new_id, 1.0)
            self.conns[in2] = ConnGene(in2, new_id, old_c.target, old_c.weight)
            
        elif choice == 'rm_node' and hiddens:
            n = random.choice(hiddens)
            del self.nodes[n.id]
            # Safely prune orphaned connections
            orphans = [i for i, c in self.conns.items() if c.source == n.id or c.target == n.id]
            for i in orphans:
                del self.conns[i]

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
                g.nodes[o] = NodeGene(o, 'output', random.uniform(-0.1, 0.1), Sigmoid)
                
            for i in self.base_inputs:
                for o in self.base_outputs:
                    innov = self.innovator.get_innov(i, o)
                    g.conns[innov] = ConnGene(innov, i, o, random.uniform(-1, 1))
            self.population.append(g)

    def evolve(self, fitness_fn, generations=100):
        for gen in range(generations):
            # 1. Evaluate
            for g in self.population:
                net = g.build_phenotype()
                g.fitness = fitness_fn(net)
                
            # 2. Sort (Highest fitness first)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            print(f"Gen {gen+1:03d} | Best Fitness: {self.population[0].fitness:.4f}")
            
            # 3. Next Generation Elitism
            next_gen = []
            elites_count = max(1, int(self.pop_size * 0.1)) # Keep top 10% exactly as they are
            next_gen.extend(self.population[:elites_count])
            
            # 4. Breed & Mutate to fill the rest
            while len(next_gen) < self.pop_size:
                # Tournament selection favoring the top 50%
                p1 = random.choice(self.population[:self.pop_size // 2])
                p2 = random.choice(self.population[:self.pop_size // 2])
                
                # Ensure p1 is the fitter parent
                if p2.fitness > p1.fitness:
                    p1, p2 = p2, p1
                    
                child = p1.crossover(p2)
                
                # 80% chance to mutate a child before entering the new generation
                if random.random() < 0.80:
                    child.mutate(self.innovator)
                    
                next_gen.append(child)
                
            self.population = next_gen
            
        return self.population[0].build_phenotype() # Return best network