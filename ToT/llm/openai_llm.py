from typing import Optional

from openai import OpenAI

from ToT.llm.base import BaseLLM
from ToT.utils import config


class OpenAILLM(BaseLLM):
    def __init__(self, client: OpenAI) -> None:
        self.client: OpenAI = client

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
            completion = self.client.chat.completions.create(
                model=config["openai_model_name"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
            response = completion.choices[0].message.content
            responses.append(response.strip())

        return responses
