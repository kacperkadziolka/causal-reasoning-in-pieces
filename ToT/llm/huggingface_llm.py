import os
from functools import lru_cache
from typing import Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Pipeline, pipeline

from ToT.llm.base import BaseLLM
from ToT.utils import config, log_time


class HuggingFaceLLM(BaseLLM):
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline: Pipeline = pipeline

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, num_samples: int = 1) -> list[str]:
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
            outputs = self.pipeline(
                messages,
                max_new_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )

            response = outputs[0]["generated_text"][-1]["content"]
            responses.append(response.strip())
