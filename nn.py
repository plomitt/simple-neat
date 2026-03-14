import math
import random

# --- 1. Math ---
class Sigmoid:
    @staticmethod
    def func(x):
        # Clip to avoid math.exp overflow for large positive/negative inputs
        if x < -700: return 0.0
        if x > 700: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def deriv(y):
        # The derivative of sigmoid is y * (1 - y), where y is the already activated output
        return y * (1.0 - y)

class ReLU:
    @staticmethod
    def func(x):
        return max(0.0, x)
    
    @staticmethod
    def deriv(y):
        # Technically needs x, but y > 0 perfectly implies x > 0 for standard ReLU
        return 1.0 if y > 0 else 0.0

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
        # The derivative of 1/2(prediction - target)^2 is simply (prediction - target)
        return prediction - target

# --- 2. Smart Connection ---
class SharedWeight:
    # Wrap float in an object so multiple connections can share the exact same reference (for convolutions)
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
        
        self.connections_in =[]
        self.connections_out = []
        self.connections_gating =[]
        self.delta = 0.0

    def forward(self):
        z = 0.0
        # z = Sum(Input * Weight * Gate_Value)
        for conn in self.connections_in:
            signal = conn.source.value * conn.weight.data
            if conn.gater: 
                signal *= conn.gater.value
            z += signal
        self.value = self.activation_cls.func(z + self.bias.data)

    def calculate_delta(self, target=None, loss_deriv_func=None):
        derivative = self.activation_cls.deriv(self.value)
        
        if target is not None:
            # Output Layer: Delta = Error * Activation_Derivative
            error = loss_deriv_func(self.value, target)
            self.delta = error * derivative
        else:
            # Hidden Layer: Delta = Sum(Weighted Outgoing Deltas) * Activation_Derivative
            total_error = 0.0
            
            # Standard backprop: accumulate error from connections where I am the source
            for conn in self.connections_out:
                eff_weight = conn.weight.data
                if conn.gater: 
                    eff_weight *= conn.gater.value
                total_error += eff_weight * conn.target.delta

            # Gater backprop: accumulate error from connections where I act as a valve multiplier
            for conn in self.connections_gating:
                total_error += conn.target.delta * conn.weight.data * conn.source.value

            self.delta = total_error * derivative

    def calculate_gradients(self):
        for conn in self.connections_in:
            input_val = conn.source.value
            if conn.gater: 
                input_val *= conn.gater.value
            
            # MAGICAL PART FIX: Accumulate gradients instead of instantly applying them!
            conn.weight.grad += self.delta * input_val 
            
        self.bias.grad += self.delta

class MaxPoolNeuron(Neuron):
    def __init__(self, label=""):
        super().__init__(Linear, label)
        self.bias.data = 0.0 
        
    def forward(self):
        if not self.connections_in: return
        
        # 1. Find the connection with the highest incoming value (the winner)
        winner = max(self.connections_in, key=lambda c: c.source.value)
        self.value = winner.source.value
        
        # 2. Dynamic Routing: Set winner's weight to 1.0, others to 0.0
        # This seamlessly routes the backward gradient ONLY to the winning pixel!
        for conn in self.connections_in:
            conn.weight.data = 1.0 if conn == winner else 0.0
            
    def calculate_gradients(self):
        pass # Pool neurons don't have trainable weights

# --- 4. Network ---
class Network:
    def __init__(self):
        self.layers =[] 

    def add_layer(self, count, activation=Sigmoid):
        new_layer =[Neuron(activation, f"L{len(self.layers)}_{i}") for i in range(count)]
        
        if self.layers:
            for prev_n in self.layers[-1]:
                for new_n in new_layer:
                    self.connect(prev_n, new_n)
                    
        self.layers.append(new_layer)

    def connect(self, source, target, gater=None, weight_obj=None):
        c = Connection(source, target, gater, weight_obj)
        source.connections_out.append(c)
        target.connections_in.append(c)
        if gater:
            gater.connections_gating.append(c)

    def predict(self, input_data):
        for i, val in enumerate(input_data):
            self.layers[0][i].value = val
            
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.forward()
                
        return [n.value for n in self.layers[-1]]

    def train(self, x, y, lr=0.1, loss_cls=MSE):
        self.predict(x)
        
        # 1. Calculate Output Deltas
        for i, n in enumerate(self.layers[-1]):
            n.calculate_delta(target=y[i], loss_deriv_func=loss_cls.derivative)
            
        # 2. Calculate Hidden Deltas
        for i in reversed(range(len(self.layers) - 1)):
            for n in self.layers[i]:
                n.calculate_delta()
        
        # 3. Accumulate Gradients (The equivalent of PyTorch's .backward())
        for layer in self.layers:
            for n in layer:
                n.calculate_gradients()
                
        # 4. Apply and Reset Gradients (The equivalent of optimizer.step() and zero_grad())
        for layer in self.layers:
            for n in layer:
                # Update Bias
                if n.bias.grad != 0.0:
                    n.bias.data -= lr * n.bias.grad
                    n.bias.grad = 0.0
                
                # Update Weights (Applies only ONCE per shared object!)
                for conn in n.connections_in:
                    if conn.weight.grad != 0.0:
                        conn.weight.data -= lr * conn.weight.grad
                        conn.weight.grad = 0.0
    
    def fit(self, X_train, Y_train, epochs=10, learning_rate=0.1, loss_cls=MSE, print_every=1):
        print(f"--- Starting Training for {epochs} Epochs ---")
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for x, y in zip(X_train, Y_train):
                self.train(x, y, learning_rate, loss_cls)
                
                predictions = [n.value for n in self.layers[-1]]
                total_loss += loss_cls.calculate(predictions, y)
                
            avg_loss = total_loss / len(X_train)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f}")
            
        print()

    def evaluate(self, X_test, Y_test, verbose=True):
        correct = 0
        for x, y in zip(X_test, Y_test):
            predictions = self.predict(x)
            
            predicted_class = predictions.index(max(predictions))
            true_class = y.index(max(y))
            
            if predicted_class == true_class:
                correct += 1
                
        accuracy = (correct / len(X_test)) * 100
        if verbose:
            print(f"Test Accuracy: {accuracy:.1f}% ({correct}/{len(X_test)} correct)")
            
        return accuracy
    
    def add_conv_filter(self, img_width, img_height, kernel_size=3, activation=ReLU):
        out_width = img_width - kernel_size + 1
        out_height = img_height - kernel_size + 1
        
        kernel_weights =[SharedWeight() for _ in range(kernel_size ** 2)]
        shared_bias = SharedWeight()
        
        new_layer =[]
        
        for y in range(out_height):
            for x in range(out_width):
                n = Neuron(activation, label=f"Conv_{x}_{y}")
                n.bias = shared_bias 
                new_layer.append(n)
                
                weight_idx = 0
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        # Calculate the 1D array index from the 2D spatial coordinates
                        src_x = x + kx
                        src_y = y + ky
                        src_index = src_y * img_width + src_x
                        
                        src_neuron = self.layers[-1][src_index]
                        # Enforce weight sharing by passing the exact same SharedWeight object 
                        self.connect(src_neuron, n, weight_obj=kernel_weights[weight_idx])
                        weight_idx += 1
                        
        self.layers.append(new_layer)
        return out_width, out_height

    def add_max_pool_layer(self, img_width, img_height, pool_size=2):
        out_width = img_width // pool_size
        out_height = img_height // pool_size
        
        new_layer =[]
        for y in range(out_height):
            for x in range(out_width):
                n = MaxPoolNeuron(label=f"Pool_{x}_{y}")
                new_layer.append(n)
                
                for py in range(pool_size):
                    for px in range(pool_size):
                        src_x = x * pool_size + px
                        src_y = y * pool_size + py
                        src_index = src_y * img_width + src_x
                        
                        src_neuron = self.layers[-1][src_index]
                        # Note: Pool connections MUST have isolated weights, as forward() dynamically overwrites them to 1.0 or 0.0
                        self.connect(src_neuron, n, weight_obj=SharedWeight(0.0))
                        
        self.layers.append(new_layer)
        return out_width, out_height