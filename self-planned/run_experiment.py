import argparse
import asyncio
import sys

from src.experiment import ExperimentRunner


async def main():
    parser = argparse.ArgumentParser(description="Run self-planned PC algorithm experiments")
    parser.add_argument("--samples", "-n", type=int, default=30,
                       help="Number of samples to test (default: 10)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name (default: auto-generated)")
    parser.add_argument("--dataset", "-d", type=str, default="../data/test_dataset.csv",
                       help="Path to dataset CSV file")

    args = parser.parse_args()

    print(f"🧪 Running experiment with {args.samples} samples")
    if args.seed:
        print(f"🎲 Using seed: {args.seed}")
    if args.name:
        print(f"📝 Experiment name: {args.name}")

    runner = ExperimentRunner(args.dataset)

    try:
        results_file = await runner.run_experiment(
            n_samples=args.samples,
            seed=args.seed,
            experiment_name=args.name
        )
        print("\n✅ Experiment completed successfully!")
        print(f"📊 Results saved to: {results_file}")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)