import argparse
import json
import logging
import random
import time
from pathlib import Path

from tqdm import tqdm

from causal_discovery.llm_client import OpenAIClient, DeepSeekClient, HuggingFaceClient, BaseLLMClient
from causal_discovery.utils import chunk_list
from shortest_path.experiment_logger import ShortestPathLogger
from shortest_path.evaluation import compare_shortest_path, aggregate_metrics, display_metrics
from shortest_path.pipeline.pipeline import ShortestPathPipeline, BatchShortestPathPipeline
from shortest_path.pipeline.stages import GraphParsingStage, DijkstraExecutionStage, ResultVerificationStage
from shortest_path.utils import load_dataset

LOGS_DIR: Path = Path(__file__).parent / "logs"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Shortest Path (Dijkstra) Pipeline with configurable backend and mode."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/nlgraph_shortest_path_main.json",
        help="Path to the NLGraph dataset JSON file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "huggingface", "deepseek"],
        default="openai",
        help="Choose the LLM backend.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "batched"],
        default="batched",
        help="Run pipeline in sequential or batched mode.",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=380,
        help="Number of experiments to run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for batch processing.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "hard"],
        default=None,
        help="Filter dataset by difficulty level. If not set, use all samples.",
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Skip the result verification stage (use only 2 stages instead of 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--retry-file",
        type=str,
        default=None,
        help="Path to a JSON file with failed sample IDs from a previous run. Only these samples will be retried.",
    )
    return parser.parse_args()


def prepare_input_samples(samples: list[dict], num_experiments: int, difficulty: str = None) -> list[dict]:
    """Prepare input samples for the pipeline."""
    # Filter by difficulty if specified
    if difficulty:
        samples = [s for s in samples if s.get("difficulty") == difficulty]
        logging.info(f"Filtered to {len(samples)} '{difficulty}' samples.")

    # Sample
    num_experiments = min(num_experiments, len(samples))
    selected = random.sample(samples, num_experiments) if num_experiments < len(samples) else samples

    input_samples = []
    for i, sample in enumerate(selected):
        input_samples.append({
            "sample_id": sample.get("id", i),
            "question": sample["question"],
            "ground_truth_answer": sample.get("answer", ""),
            "ground_truth_weight": sample.get("total_weight"),
            "difficulty": sample.get("difficulty", "unknown"),
            "num_nodes": sample.get("num_nodes"),
            "num_edges": sample.get("num_edges"),
        })

    logging.info(f"Prepared {len(input_samples)} input samples for the pipeline.")
    return input_samples


def create_client(backend: str, batch_size: int) -> BaseLLMClient:
    if backend == "openai":
        client = OpenAIClient(model_id="o3-mini", concurrency=batch_size)
    elif backend == "huggingface":
        client = HuggingFaceClient(max_new_tokens=8192, batch_size=batch_size)
    else:
        client = DeepSeekClient(concurrency=batch_size)
    logging.info(f"Using {backend} backend for the pipeline.")
    return client


def post_process_logs(results: list[dict], input_samples: list[dict]) -> None:
    """Compute and display metrics from pipeline results."""
    eval_results = []
    difficulties = []

    for result, sample in zip(results, input_samples):
        ground_truth_weight = sample.get("ground_truth_weight")
        question = sample.get("question", "")
        difficulty = sample.get("difficulty", "unknown")

        comparison = compare_shortest_path(
            result={"path": result.get("path"), "total_weight": result.get("total_weight")},
            ground_truth_weight=ground_truth_weight,
            question=question,
        )
        eval_results.append(comparison)
        difficulties.append(difficulty)

    metrics = aggregate_metrics(eval_results, difficulties)
    display_metrics(metrics)


def main() -> None:
    args = parse_arguments()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    random.seed(args.seed)
    logging.info(f"Random seed: {args.seed}")

    # Load dataset
    all_samples = load_dataset(args.input_file)

    if args.retry_file:
        with open(args.retry_file, 'r') as f:
            retry_data = json.load(f)
        retry_ids = set(retry_data["failed_sample_ids"])
        input_samples = prepare_input_samples(all_samples, len(all_samples), args.difficulty)
        input_samples = [s for s in input_samples if s["sample_id"] in retry_ids]
        logging.info(f"Retrying {len(input_samples)} failed samples from {args.retry_file}")
    else:
        input_samples = prepare_input_samples(all_samples, args.num_experiments, args.difficulty)

    # Create LLM client
    client = create_client(args.backend, args.batch_size)

    # Create stages
    stages = [
        GraphParsingStage(client=client),
        DijkstraExecutionStage(client=client),
    ]
    if not args.skip_verification:
        stages.append(ResultVerificationStage(client=client))

    stage_names = [s.__class__.__name__ for s in stages]
    print(f"\nPipeline stages ({len(stages)}): {' -> '.join(stage_names)}")
    if args.skip_verification:
        print("(Verification stage SKIPPED)")

    # Create pipeline
    job_id = Path(args.input_file).stem
    if args.difficulty:
        job_id += f"_{args.difficulty}"
    logger = ShortestPathLogger(LOGS_DIR, job_id)
    pipeline = ShortestPathPipeline(stages=stages, logger=logger)

    results = []
    successful_samples = []
    failed_ids = []
    start_time = time.time()

    if args.mode == "batched":
        logging.info("Running pipeline in batched mode.")
        batch_pipeline = BatchShortestPathPipeline(
            pipeline=pipeline,
            batch_size=args.batch_size,
        )
        results, failed_ids = batch_pipeline.run_batch(input_samples)
        successful_samples = [s for s in input_samples if s["sample_id"] not in failed_ids]
    else:
        logging.info("Running pipeline in sequential mode.")
        for sample in tqdm(input_samples, desc="Processing samples"):
            try:
                result = pipeline.run(sample)
                results.append(result)
                successful_samples.append(sample)
            except Exception as e:
                failed_ids.append(sample["sample_id"])
                logging.error(f"Error processing sample {sample['sample_id']}: {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/sample)")

    # Post-process and display metrics
    if results:
        post_process_logs(results, successful_samples)

    # Token usage summary
    total_input = sum(r.get("token_usage", {}).get("input_tokens", 0) for r in results)
    total_output = sum(r.get("token_usage", {}).get("output_tokens", 0) for r in results)
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
    n_run = len(results)
    if n_run > 0:
        print(f"\nToken Usage: {total_input:,} input + {total_output:,} output = {total_tokens:,} total "
              f"(avg {total_tokens // n_run:,}/sample)")

    # Collect samples with extraction failures (None path and weight)
    extraction_failed_ids = [
        r.get("sample_id") for r in results
        if r.get("path") is None and r.get("total_weight") is None
    ]
    all_failed_ids = list(set(failed_ids + extraction_failed_ids))

    if all_failed_ids:
        failed_file = LOGS_DIR / f"failed_samples_{time.strftime('%Y%m%d_%H%M%S')}.json"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(failed_file, 'w') as f:
            json.dump({"failed_sample_ids": all_failed_ids, "count": len(all_failed_ids)}, f, indent=2)
        print(f"\nFailed samples ({len(all_failed_ids)}): {all_failed_ids}")
        print(f"Retry with: python -m shortest_path.main --retry-file {failed_file}")
    else:
        print("\nAll samples completed successfully.")

    print(f"\nResults saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
