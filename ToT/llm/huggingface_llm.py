from typing import Optional

from transformers import Pipeline

from ToT.llm.base import BaseLLM
from ToT.utils import config


class HuggingFaceLLM(BaseLLM):
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline: Pipeline = pipeline

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, num_samples: int = 1) -> list[str]:
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

        # Create batch of identical messages for parallel generation
        batch_messages = [messages] * num_samples

        # Process all samples in one batch
        outputs = self.pipeline(
            batch_messages,
            max_new_tokens=config["max_tokens"],
            temperature=config["temperature"],
            batch_size=num_samples
        )

        # Extract responses from batch output
        responses = []
        for output in outputs:
            response = output[0]["generated_text"][-1]["content"]
            responses.append(response.strip())

        return responses
