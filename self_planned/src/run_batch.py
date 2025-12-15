import argparse
import asyncio
import json
from pathlib import Path

from execute.batch_experiments import BatchExperimentRunner, ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch experiments")

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
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
        default=2,
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

    parser.add_argument(
        "--sample-indices", "-si",
        type=str,
        help="Path to JSON file containing specific sample indices to run",
        default="indices/sample_indices_batch_exp_20251125_222142_results.json"
    )

    parser.add_argument(
        "--sequential-generation",
        action="store_true",
        default=True,  # Sequential mode is now the default
        help="Generate stage prompts sequentially instead of batch (only with --multi-agent-planner)"
    )

    parser.add_argument(
        "--multi-agent-planner",
        action="store_true",
        default=True,  # Multi-agent planner is now the default
        help="Use MultiAgentPlanner instead of IterativePlanner"
    )

    parser.add_argument(
        "--use-plan-caching",
        action="store_true",
        default=True,
        help="Enable plan caching (regenerate plan every N samples) [default: enabled]"
    )

    parser.add_argument(
        "--no-plan-caching",
        action="store_false",
        dest="use_plan_caching",
        help="Disable plan caching (regenerate plan for every sample)"
    )

    parser.add_argument(
        "--plan-cache-size",
        type=int,
        default=None,
        help="Number of samples sharing one plan (default: same as --max-concurrent)"
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

    # Load sample indices if provided
    sample_indices = None
    if args.sample_indices:
        sample_indices_path = Path(args.sample_indices)
        if not sample_indices_path.exists():
            print(f"❌ Error: Sample indices file not found: {args.sample_indices}")
            return

        try:
            with open(sample_indices_path, 'r') as f:
                indices_data = json.load(f)
                sample_indices = indices_data.get('sample_indices')
                if sample_indices is None:
                    print(f"❌ Error: No 'sample_indices' key found in {args.sample_indices}")
                    return
                print(f"📋 Loaded {len(sample_indices)} sample indices from {args.sample_indices}")
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in sample indices file: {e}")
            return
        except Exception as e:
            print(f"❌ Error: Failed to load sample indices file: {e}")
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
        save_summary=not args.no_summary,
        sample_indices=sample_indices,
        use_sequential_generation=args.sequential_generation,
        use_multi_agent_planner=args.multi_agent_planner,
        use_plan_caching=args.use_plan_caching,
        plan_cache_size=args.plan_cache_size
    )

    print("🔧 Experiment Configuration:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max concurrent: {config.max_concurrent}")
    print(f"   Dataset: {config.dataset_path}")
    print(f"   Output dir: {config.output_dir}")
    print(f"   Random seed: {config.random_seed}")
    print(f"   Save individual: {config.save_individual_results}")
    print(f"   Save summary: {config.save_summary}")
    print(f"   Multi-agent planner: {config.use_multi_agent_planner}")
    print(f"   Sequential generation: {config.use_sequential_generation}")
    print(f"   Plan caching: {config.use_plan_caching}")
    if config.use_plan_caching:
        cache_size = config.plan_cache_size or config.max_concurrent
        print(f"   Plan cache size: {cache_size} samples per plan")

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