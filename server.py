import asyncio
import websockets
import json
import random

# Import our custom logic
from snake_game import SnakeGame
from nn_graph import Linear, Sigmoid
from neuroevolution import Genome, Innovator, NodeGene, ConnGene

# Visualization functions
from viz import save_network_to_json
from viz2 import visualize_network

POP_SIZE = 100
GRID_SIZE = 15
TARGET_SCORE = 100 # Stops evolution when a snake eats N apples
TICK_SPEED = 0.01

async def main():
    # 1. Initialize Genome Population
    innovator = Innovator()
    population =[]
    base_inputs = [innovator.get_node_id() for _ in range(6)]
    base_outputs = [innovator.get_node_id() for _ in range(2)]

    for _ in range(POP_SIZE):
        g = Genome()
        for i in base_inputs: g.nodes[i] = NodeGene(i, 'input', 0.0, Linear)
        for o in base_outputs: g.nodes[o] = NodeGene(o, 'output', random.uniform(-1, 1), Sigmoid)
        for i in base_inputs:
            for o in base_outputs:
                innov = innovator.get_innov(i, o)
                g.conns[innov] = ConnGene(innov, i, o, random.uniform(-1, 1))
        population.append(g)

    # 2. WebSocket Broadcasting Setup
    clients = set()
    async def handler(websocket):
        clients.add(websocket)
        try: await websocket.wait_closed()
        finally: clients.remove(websocket)

    # Start WebSocket Server on Port 8765
    server = await websockets.serve(handler, "localhost", 8765)
    print("Server started at ws://localhost:8765")
    print("Open index.html in your browser to watch evolution!")

    # 3. The Master Game/Evolution Loop
    generation = 1
    best_all_time = 0
    best_all_time_net = None

    while True:
        games =[SnakeGame(GRID_SIZE, seed=generation) for _ in range(POP_SIZE)]
        nets =[g.build_phenotype() for g in population]

        # Tick loop until all snakes are dead
        while any(not game.dead for game in games):
            gen_best = 0
            state_data =[]

            for i, (game, net) in enumerate(zip(games, nets)):
                if not game.dead:
                    # Clear memory, get vision, predict, step
                    for node in net.hiddens + net.outputs: node.value = 0.0
                    vision = game.get_vision()
                    action = net.predict(vision)
                    game.step(action[0], action[1])

                if game.score > gen_best: gen_best = game.score
                if game.score > best_all_time: 
                    best_all_time = game.score
                    best_all_time_net = net

                state_data.append({
                    "id": i,
                    "snake": game.snake,
                    "apple": game.apple,
                    "score": game.score,
                    "dead": game.dead
                })

            # Blast state to Web Browser
            if clients:
                payload = json.dumps({
                    "generation": generation,
                    "best_current": gen_best,
                    "best_all_time": best_all_time,
                    "grid_size": GRID_SIZE,
                    "games": state_data
                })
                websockets.broadcast(clients, payload)

            await asyncio.sleep(TICK_SPEED) # Default Tick Speed

        # --- GENERATION ENDED: DO NEAT EVOLUTION ---
        print(f"Gen {generation:03d} | Best Score: {gen_best} | All Time: {best_all_time}")
        
        if best_all_time >= TARGET_SCORE:
            print("Target reached! Saving and visualizing...")
            save_network_to_json(best_all_time_net, "snake_best_net/snake_champion.json")
            visualize_network(best_all_time_net)
            break # Exit Loop

        # Calculate Fitness (Apples dominate, size is penalized)
        for g, game in zip(population, games):
            size_penalty = (len(g.nodes) + len(g.conns)) * 0.01
            current_fitness = game.score - size_penalty
            
            g.fitness_history.append(current_fitness)
            if len(g.fitness_history) > 5:
                g.fitness_history.pop(0) # Keep only the last 5 generations
                
            g.effective_fitness = sum(g.fitness_history) / len(g.fitness_history)

        # Sort using the averaged effective_fitness
        population.sort(key=lambda x: x.effective_fitness, reverse=True)

        # Breed next generation (Elitism + Innovation Immunity + Crossover + Mutation)
        next_gen =[]
        elites = max(1, int(POP_SIZE * 0.1))
        next_gen.extend(population[:elites])
        
        for g in population[elites:]:
            if g.is_innovative and len(next_gen) < POP_SIZE:
                next_gen.append(g)
                
        for g in next_gen: g.is_innovative = False

        while len(next_gen) < POP_SIZE:
            p1 = random.choice(population[:POP_SIZE//2])
            p2 = random.choice(population[:POP_SIZE//2])
            if p2.fitness > p1.fitness: p1, p2 = p2, p1
            child = p1.crossover(p2)
            if random.random() < 0.8: child.mutate(innovator)
            next_gen.append(child)

        population = next_gen
        generation += 1

if __name__ == "__main__":
    asyncio.run(main())