import argparse
import logging
import os
from datetime import datetime

import asyncio
from dotenv import load_dotenv

from openai_backend import run_experiments_openai_async
from utils import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shortest path experiments with OpenAI reasoning models on the NLGraph dataset."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai"],
        help="Which backend to use (currently only openai)",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=None,
        help="Number of experiments to run (default: all samples)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="../data/nlgraph_shortest_path_main.json",
        help="Path to the NLGraph JSON dataset",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="direct_prompt",
        choices=["direct_prompt", "cot_prompt", "algorithm_prompt", "bag_prompt"],
        help="Which prompt strategy to use",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "hard"],
        help="Filter by difficulty level (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    input_filename = os.path.basename(args.input_file)
    file_input_name = os.path.splitext(input_filename)[0]
    log_file = os.path.join(log_dir, f"{args.backend}_{timestamp}_{file_input_name}.csv")

    # Load dataset (JSON)
    samples = load_dataset(args.input_file)

    # Filter by difficulty if specified
    if args.difficulty:
        samples = [s for s in samples if s["difficulty"] == args.difficulty]
        logging.info(f"Filtered to {len(samples)} {args.difficulty} samples")

    num_experiments = args.num_experiments or len(samples)
    logging.info(f"Running {num_experiments} experiments with prompt_type={args.prompt_type}")

    if args.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        asyncio.run(
            run_experiments_openai_async(
                samples=samples,
                num_experiments=num_experiments,
                log_file=log_file,
                api_key=api_key,
                prompt_type=args.prompt_type,
            )
        )


if __name__ == "__main__":
    main()
