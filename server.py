import time
import asyncio, json, random, websockets
from snake_game import SnakeGame
from nn_graph import Linear, Sigmoid
from neuroevolution import Genome, Innovator, NodeGene, ConnGene

# Import your save and viz functions!
from viz import save_network_to_json
from viz2 import visualize_network

config = {
    "pop_size": 100, "grid_size": 15, "target_score": 100, "tick_speed": 0.01,
    "prob_mutate_weight": 0.8, "prob_mutate_bias": 0.8, "prob_mutate_act": 0.1,
    "prob_add_node": 0.4, "prob_add_conn": 0.6, "prob_rm_node": 0.1,
    "prob_rm_conn": 0.1, "prob_swap": 0.05, "prob_diversity_injection": 0.3
}
state = {"paused": True, "reset": False}
clients, logs = set(),[]

def log(msg):
    logs.append(msg)
    if len(logs) > 30: logs.pop(0)
    websockets.broadcast(clients, json.dumps({"type": "log", "msg": msg}))
    print(msg)

async def handler(ws):
    global best_all_time_net # Need to access the global best net
    clients.add(ws)
    await ws.send(json.dumps({"type": "config", "cfg": config}))
    for m in logs: await ws.send(json.dumps({"type": "log", "msg": m}))
    try:
        async for msg in ws:
            d = json.loads(msg)
            if d["cmd"] == "play": state["paused"] = False; log("▶ Resumed")
            elif d["cmd"] == "pause": state["paused"] = True; log("⏸ Paused")
            elif d["cmd"] == "reset": state["reset"] = True; log("🔄 Reset")
            elif d["cmd"] == "update": config[d["key"]] = float(d["val"]); log(f"⚙ {d['key']} = {d['val']}")
            elif d["cmd"] == "save_best":
                if best_all_time_net:
                    fname = f"snake_best_{int(time.time())}.json"
                    save_network_to_json(best_all_time_net, fname)
                    log(f"💾 Saved: {fname}")
                else:
                    log("⚠ No best net to save yet!")
    finally: clients.remove(ws)

async def evolution_loop():
    global best_all_time_net

    while True:
        state["reset"] = False
        inv = Innovator()
        ins, outs =[inv.get_node_id() for _ in range(6)],[inv.get_node_id() for _ in range(2)]
        pop =[]
        
        # Init Population
        for _ in range(int(config["pop_size"])):
            g = Genome()
            for i in ins: g.nodes[i] = NodeGene(i, 'input', 0.0, Linear)
            for o in outs: g.nodes[o] = NodeGene(o, 'output', random.uniform(-1, 1), Sigmoid)
            for i in ins:
                for o in outs:
                    innov = inv.get_innov(i, o)
                    g.conns[innov] = ConnGene(innov, i, o, random.uniform(-1, 1))
            pop.append(g)

        gen, best_all, best_all_time_net = 1, 0, None

        while not state["reset"]:
            while state["paused"]: await asyncio.sleep(0.1)
            if state["reset"]: break

            games =[SnakeGame(int(config["grid_size"]), seed=gen) for _ in range(len(pop))]
            nets = [g.build_phenotype() for g in pop]

            for nt in nets:
                for n in nt.hiddens + nt.outputs: n.value = 0.0

            # Tick loop
            while any(not gm.dead for gm in games):
                if state["paused"] or state["reset"]: break
                
                state_data, gen_best =[], 0
                for i, (gm, nt) in enumerate(zip(games, nets)):
                    if not gm.dead:
                        act = nt.predict(gm.get_vision())
                        gm.step(act[0], act[1])
                    
                    gen_best = max(gen_best, gm.score)
                    if gm.score > best_all: best_all, best_all_time_net = gm.score, nt
                    state_data.append({"id": i, "snake": gm.snake, "apple": gm.apple, "score": gm.score, "dead": gm.dead})

                websockets.broadcast(clients, json.dumps({
                    "type": "state", "gen": gen, "best": gen_best, "all": best_all,
                    "grid": int(config["grid_size"]), "games": state_data
                }))
                await asyncio.sleep(config["tick_speed"])

            if state["reset"] or state["paused"]: continue

            log(f"Gen {gen} Ended | Best: {gen_best}")
            
            # --- THE TARGET WIN CONDITION ---
            if best_all >= config["target_score"]:
                log(f"Target {config['target_score']} Reached! Saving & Visualizing...")
                save_network_to_json(best_all_time_net, "snake_champion.json")
                visualize_network("snake_champion.json")
                state["paused"] = True  # Auto-pause so you can admire the winner
                continue 

            # Calculate Fitness (Moving average penalty)
            for g, gm in zip(pop, games):
                if not hasattr(g, 'hist'): g.hist =[]
                g.hist.append(gm.score)
                g.fit = sum(g.hist[-5:]) / len(g.hist[-5:])

            # Sort and Breed
            pop.sort(key=lambda x: x.fit, reverse=True)
            nxt = pop[:max(1, int(len(pop) * 0.05))] # Elites
            nxt +=[g for g in pop[len(nxt):] if getattr(g, 'is_innovative', False)][:len(pop)-len(nxt)]
            for g in nxt: g.is_innovative = False

            muts = {}
            while len(nxt) < int(config["pop_size"]):
                p1, p2 = random.sample(pop[:len(pop)//2], 2)
                child = (p1 if p1.fit > p2.fit else p2).crossover(p2 if p1.fit > p2.fit else p1, config)
                if random.random() < 0.8:
                    for m in child.mutate(inv, config): muts[m] = muts.get(m, 0) + 1
                nxt.append(child)

            pop = nxt
            gen += 1
            mut_str = ", ".join([f"{k}:{v}" for k, v in muts.items()])
            log(f"Bred {len(pop)}. Muts: {mut_str or 'None'}")

async def main():
    await websockets.serve(handler, "localhost", 8765)
    log("Server Started ws://localhost:8765")
    await evolution_loop()

if __name__ == "__main__": asyncio.run(main())