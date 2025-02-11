from abc import ABC
from typing import Optional


class BaseLLM(ABC):
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, num_samples: int = 1) -> list[str]:
        """
        Calls the LLM with a given prompt and returns a list of responses depends on the number of samples.
        If system_prompt is provided, it is sent along with the user_prompt; otherwise, only the user_prompt is sent.
        """
        pass
