from neuroevolution import EvolutionManager
from viz import save_network_to_json
from viz2 import visualize_network

XOR_DATA = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

def xor_fitness(network):
    # Calculate fitness as: 4.0 - Total Error
    # A perfect network will have a fitness of 4.0
    error = 0.0
    for x, y in XOR_DATA:
        prediction = network.predict(x)[0]
        error += (y[0] - prediction) ** 2
    return 4.0 - error

print("Starting Neuroevolution for XOR...")
evo = EvolutionManager(input_size=2, output_size=1, pop_size=200)

# Run evolution
best_net = evo.evolve(xor_fitness, generations=500, fitness_threshold=3.99)

print("\n--- Evolution Complete ---")
print(f"Best Network has {len(best_net.hiddens)} hidden nodes.")

print("\nTest Results:")
for x, y in XOR_DATA:
    pred = best_net.predict(x)[0]
    print(f"In: {x} -> Out: {pred:.4f} (Target {y[0]})")

save_network_to_json(best_net, "best_net/xor_champion.json")
visualize_network(best_net)