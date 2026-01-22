import argparse
import logging
import os
from datetime import datetime

import asyncio
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer

from deepseek_async_backend import run_experiments_deepseek_async
from huggingface_backend import run_experiments_hf
from openai_backend import run_experiments_openai_async


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
        choices=["openai", "huggingface", "deepseek"],
        help="Which backend to use: openai, huggingface, or deepseek"
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="Number of experiments to run"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the split csv file",
        default="../data/test_dataset.csv"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="single_stage_prompt",
        choices=["single_stage_prompt", "minimal_prompt"],
        help="Which prompt to use: single_stage_prompt (detailed) or minimal_prompt (simple)"
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    input_filename = os.path.basename(args.input_file)
    file_input_name = os.path.splitext(input_filename)[0]
    log_file = os.path.join(log_dir, f"{args.backend}_{timestamp}_{file_input_name}.csv")

    df = pd.read_csv(args.input_file)

    if args.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        asyncio.run(
            run_experiments_openai_async(
                df=df,
                num_experiments=args.num_experiments,
                log_file=log_file,
                api_key=api_key,
                prompt_type=args.prompt_type
            )
        )

    elif args.backend == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

        asyncio.run(
            run_experiments_deepseek_async(
                df=df,
                log_file=log_file,
                api_key=api_key,
                num_experiments=args.num_experiments,
                prompt_type=args.prompt_type
            )
        )

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
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"Failed to load Hugging Face model: {e}")
            return
        run_experiments_hf(hf_pipeline, df, args.num_experiments, log_file, tokenizer, prompt_type=args.prompt_type)


if __name__ == "__main__":
    main()
