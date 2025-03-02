import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

config = {
    "max_tokens": 2048,
    "temperature": 0.1,
}

class GroqLLM:
    def __init__(self, client: Groq) -> None:
        self.client: Groq = client

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
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
            response = completion.choices[0].message.content
            responses.append(response.strip())

        return responses


@lru_cache
def get_test_model() -> GroqLLM:
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    return GroqLLM(Groq(api_key=api_key))
