import argparse
import asyncio
from pathlib import Path

from batch_experiments import BatchExperimentRunner, ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch experiments")

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=2,
        help="Number of experiments to run (max: dataset size)"
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="../data/test_dataset.csv",
        help="Path to the dataset CSV file"
    )

    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Experiment name (default: auto-generated with timestamp)"
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed for reproducible sampling"
    )

    parser.add_argument(
        "--max-concurrent", "-c",
        type=int,
        default=10,
        help="Maximum number of concurrent experiments [default: 10]"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="experiments",
        help="Output directory for results"
    )

    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Don't save individual results CSV"
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't save summary JSON"
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found: {args.dataset}")
        print("📁 Make sure the dataset exists or provide correct path with --dataset")
        return

    # Create configuration
    config = ExperimentConfig(
        batch_size=args.batch_size,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        experiment_name=args.name,
        random_seed=args.seed,
        max_concurrent=args.max_concurrent,
        save_individual_results=not args.no_individual,
        save_summary=not args.no_summary
    )

    print("🔧 Experiment Configuration:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max concurrent: {config.max_concurrent}")
    print(f"   Dataset: {config.dataset_path}")
    print(f"   Output dir: {config.output_dir}")
    print(f"   Random seed: {config.random_seed}")
    print(f"   Save individual: {config.save_individual_results}")
    print(f"   Save summary: {config.save_summary}")

    # Run experiments
    runner = BatchExperimentRunner(config)
    batch_results = await runner.run_batch()

    # Print summary
    runner.print_summary(batch_results)

    # Save results
    if config.save_individual_results or config.save_summary:
        results_file, summary_file = runner.save_results(batch_results)
        print("\n💾 Results saved:")
        if results_file:
            print(f"   📄 Details: {results_file}")
        if summary_file:
            print(f"   📋 Summary: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())