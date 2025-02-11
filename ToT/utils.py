import json
import os
from functools import lru_cache
from typing import Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI


def load_config(filepath: str = "config.json") -> dict[str, any]:
    """
    Load the configuration from a JSON file.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def load_prompts(filepath: str = "prompts.yaml") -> dict[str, str]:
    """
    Load the prompts from a YAML file.
    """
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


prompts = load_prompts()
config = load_config()

log_directory = config.get("log_directory")
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


def call_llm(client: OpenAI, user_prompt: str, system_prompt: Optional[str] = None, num_samples: int = 1) -> list[str]:
    """
    Calls the LLM with a given prompt and returns a list of responses depends on the number of samples.
    If system_prompt is provided, it is sent along with the user_prompt; otherwise, only the user_prompt is sent.
    """
    responses = []

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "user",
        "content": user_prompt
    })

    for _ in range(num_samples):
        completion = client.chat.completions.create(
            model=config["model_name"],
            messages=messages,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        response = completion.choices[0].message.content
        responses.append(response.strip())

    return responses


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
