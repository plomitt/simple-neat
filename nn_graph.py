import math
import random

# --- 1. Math ---
class Sigmoid:
    @staticmethod
    def func(x):
        if x < -700: return 0.0
        if x > 700: return 1.0
        return 1.0 / (1.0 + math.exp(-x))
    @staticmethod
    def deriv(y): return y * (1.0 - y)

class ReLU:
    @staticmethod
    def func(x): return max(0.0, x)
    @staticmethod
    def deriv(y): return 1.0 if y > 0 else 0.0

class Tanh:
    @staticmethod
    def func(x): return math.tanh(x)
    @staticmethod
    def deriv(y): return 1.0 - y**2

class Linear:
    @staticmethod
    def func(x): return x
    @staticmethod
    def deriv(y): return 1.0

class MSE:
    @staticmethod
    def calculate(predictions, targets):
        return sum((t - p)**2 for t, p in zip(targets, predictions)) / len(targets)
    @staticmethod
    def derivative(prediction, target):
        return prediction - target

# --- 2. Smart Connection ---
class SharedWeight:
    def __init__(self, value=None):
        self.data = random.uniform(-0.1, 0.1) if value is None else value
        self.grad = 0.0

class Connection:
    def __init__(self, source, target, gater=None, weight_obj=None):
        self.source = source
        self.target = target
        self.gater = gater
        self.weight = weight_obj if weight_obj else SharedWeight()

# --- 3. Neuron ---
class Neuron:
    def __init__(self, activation_cls=Sigmoid, label=""):
        self.label = label
        self.activation_cls = activation_cls
        self.value = 0.0
        self.bias = SharedWeight() 
        
        self.connections_in = []
        self.connections_out =[]
        self.connections_gating =[]
        self.delta = 0.0

    def forward(self):
        z = 0.0
        for conn in self.connections_in:
            signal = conn.source.value * conn.weight.data
            if conn.gater: 
                signal *= conn.gater.value
            z += signal
        self.value = self.activation_cls.func(z + self.bias.data)

    def calculate_delta(self, target=None, loss_deriv_func=None):
        derivative = self.activation_cls.deriv(self.value)
        if target is not None:
            error = loss_deriv_func(self.value, target)
            self.delta = error * derivative
        else:
            total_error = 0.0
            for conn in self.connections_out:
                eff_weight = conn.weight.data
                if conn.gater: eff_weight *= conn.gater.value
                total_error += eff_weight * conn.target.delta
            for conn in self.connections_gating:
                total_error += conn.target.delta * conn.weight.data * conn.source.value
            self.delta = total_error * derivative

    def calculate_gradients(self):
        for conn in self.connections_in:
            input_val = conn.source.value
            if conn.gater: input_val *= conn.gater.value
            conn.weight.grad += self.delta * input_val 
        self.bias.grad += self.delta

# --- 4. Network ---
class Network:
    def __init__(self):
        # Splitting lists ensures correct mathematical order for Feed-Forward/Backprop
        self.inputs = []
        self.hiddens = []
        self.outputs = []

    @property
    def all_nodes(self):
        return self.inputs + self.hiddens + self.outputs

    # --- Graph Mutation Primitives ---
    def add_node(self, node, node_type="hidden"):
        """node_type can be 'input', 'hidden', or 'output'"""
        if node_type == "input": self.inputs.append(node)
        elif node_type == "output": self.outputs.append(node)
        else: self.hiddens.append(node)

    def remove_node(self, node):
        """Removes a node and cleanly severs all its connections."""
        if node in self.inputs: self.inputs.remove(node)
        elif node in self.outputs: self.outputs.remove(node)
        elif node in self.hiddens: self.hiddens.remove(node)
        
        # Sever connections safely
        for conn in list(node.connections_in):
            self.remove_connection(conn.source, node)
        for conn in list(node.connections_out):
            self.remove_connection(node, conn.target)
        for conn in list(node.connections_gating):
            conn.gater = None
            node.connections_gating.remove(conn)

    def add_connection(self, source, target, gater=None, weight_val=None):
        weight_obj = SharedWeight(weight_val) if weight_val is not None else SharedWeight()
        c = Connection(source, target, gater, weight_obj)
        source.connections_out.append(c)
        target.connections_in.append(c)
        if gater: gater.connections_gating.append(c)
        return c

    def remove_connection(self, source, target):
        for c in source.connections_out:
            if c.target == target:
                source.connections_out.remove(c)
                target.connections_in.remove(c)
                if c.gater: c.gater.connections_gating.remove(c)
                break

    # --- Property Modifiers ---
    def set_activation(self, node, activation_cls):
        node.activation_cls = activation_cls

    def set_bias(self, node, bias_val):
        node.bias.data = bias_val

    def set_weight(self, source, target, weight_val):
        for c in source.connections_out:
            if c.target == target:
                c.weight.data = weight_val
                break

    # --- Execution Engine ---
    def predict(self, input_data):
        # 1. Load data into sensors (Input nodes don't 'fire', they just hold values)
        for i, val in enumerate(input_data):
            self.inputs[i].value = val
            
        # 2. Forward Propagate strictly from Hiddens -> Outputs
        for node in self.hiddens: node.forward()
        for node in self.outputs: node.forward()
                
        return[n.value for n in self.outputs]

    def train(self, x, y, lr=0.1, loss_cls=MSE):
        self.predict(x)
        
        # 1. Deltas: Backwards from Outputs -> Hiddens
        for i, n in enumerate(self.outputs):
            n.calculate_delta(target=y[i], loss_deriv_func=loss_cls.derivative)
        for n in reversed(self.hiddens):
            n.calculate_delta()
        
        # 2. Gradients
        for n in self.all_nodes: n.calculate_gradients()
                
        # 3. Apply & Reset Gradients
        for n in self.all_nodes:
            if n.bias.grad != 0.0:
                n.bias.data -= lr * n.bias.grad
                n.bias.grad = 0.0
            for conn in n.connections_in:
                if conn.weight.grad != 0.0:
                    conn.weight.data -= lr * conn.weight.grad
                    conn.weight.grad = 0.0
    
    def fit(self, X_train, Y_train, epochs=10, learning_rate=0.1, loss_cls=MSE, print_every=1):
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in zip(X_train, Y_train):
                self.train(x, y, learning_rate, loss_cls)
                predictions =[n.value for n in self.outputs]
                total_loss += loss_cls.calculate(predictions, y)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(X_train):.4f}")

    def evaluate(self, X_test, Y_test, verbose=True):
        correct = sum(1 for x, y in zip(X_test, Y_test) 
                      if self.predict(x).index(max(self.predict(x))) == y.index(max(y)))
        accuracy = (correct / len(X_test)) * 100
        if verbose: print(f"Test Accuracy: {accuracy:.1f}%")
        return accuracy