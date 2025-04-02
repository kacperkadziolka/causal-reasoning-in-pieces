import argparse
import logging
import os
from datetime import datetime

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import pipeline

from huggingface_backend import run_experiments_hf
from openai_backend import run_experiments_openai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal discovery experiments with configurable backend, dataset, and number of experiments."
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
        choices=["openai", "huggingface"],
        help="Which backend to use: openai or huggingface"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="normal",
        choices=["normal", "balanced"],
        help="Which dataset to use: normal or balanced"
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="Number of experiments to run"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_logs_{args.backend}_{args.num_experiments}exp_{timestamp}.csv")

    if args.dataset == "normal":
        dataset_file_path = "data/test_dataset_unbalanced.csv"
    else:
        dataset_file_path = "data/test_dataset_balanced.csv"
    df = pd.read_csv(dataset_file_path)

    if args.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        client = OpenAI(api_key=api_key)
        run_experiments_openai(client, df, args.num_experiments, log_file)

    elif args.backend == "huggingface":
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables.")

        login(token=hf_token)
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

        try:
            hf_pipeline = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            print(f"Failed to load Hugging Face model: {e}")
            return
        run_experiments_hf(hf_pipeline, df, args.num_experiments, log_file)


if __name__ == "__main__":
    main()
