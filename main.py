"""Main entry: CLI runs training then single simulation, or launches GUI."""
import argparse
from .simulation import SAURSimulation, SimParams
from .rl_agent import QLearningAgent


def main():
    parser = argparse.ArgumentParser(description="SAUR PSP Simulation")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--train", type=int, default=0,
                        help="Train RL for N episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-time", type=float, default=600.0)
    args = parser.parse_args()

    if args.gui:
        from .gui import launch_gui
        launch_gui()
        return

    if args.train > 0:
        print(f"Training RL agent for {args.train} episodes...")
        agent = SAURSimulation.run_training(args.train, seed=args.seed,
                                            sim_time=args.sim_time)
        agent.save("saur_rl_trained.json")
        print(f"Agent saved. Final epsilon={agent.epsilon:.3f}")
    else:
        p = SimParams(sim_time=args.sim_time, seed=args.seed, training=False)
        sim = SAURSimulation(params=p)
        results = sim.run()
        print("\n=== Simulation Results ===")
        for k, v in results.items():
            if k != "chain_summary":
                print(f"  {k}: {v}")
        print()
        print(results.get("chain_summary", ""))
