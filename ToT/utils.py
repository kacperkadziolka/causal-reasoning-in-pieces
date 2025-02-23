import json
import os
import time
from functools import lru_cache

import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import Pipeline, pipeline


def log_time(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def load_config(filepath: str = "resources/config.json") -> dict[str, any]:
    """
    Load the configuration from a JSON file.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def load_prompts(filepath: str = "resources/prompts.yaml") -> dict[str, str]:
    """
    Load the prompts from a YAML file.
    """
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


prompts = load_prompts()
evaluate_prompts = load_prompts("resources/evaluate_prompts.yaml")
config = load_config()

log_directory = config.get("log_directory")
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


@lru_cache
def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client.
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")

    return OpenAI(api_key=api_key)


@lru_cache()
def get_huggingface_pipeline() -> Pipeline:
    """
    Create a Hugging Face pipeline.
    """
    load_dotenv()
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    log_time("Creating a huggingface pipeline object")
    log_time(f"Torch: {torch.__version__}")
    log_time(f"CUDA: {torch.version.cuda}")

    log_time("Logging into Hugging Face hub...")
    login(token=huggingface_token)

    log_time("Loading the Llama model...")
    model_id: str = config["hf_model_name"]
    cache_dir = config["hf_cache_dir"]
    cache_kwargs = dict(cache_dir=cache_dir, local_files_only=True)

    return pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=cache_kwargs,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
