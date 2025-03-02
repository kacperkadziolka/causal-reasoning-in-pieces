import logging
from typing import Any

import yaml

from fail_experiments.testing_sandbox.conf_groq import GroqLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_prompts(file_path: str) -> dict[str, Any]:
    """Load prompts from YAML file."""
    with open(file_path, 'r') as file:
        prompts = yaml.safe_load(file)
    return prompts


class CorrectionPipeline:
    def __init__(self, llm: GroqLLM, prompts_file: str = "prompts.yaml"):
        self.llm = llm
        self.prompts = load_prompts(prompts_file)

        # Common errors from dataset
        self.common_missing_edges = [('D', 'E'), ('C', 'D')]
        self.common_extra_edges = [('B', 'E'), ('A', 'C')]

    def verify_edge(self, premise: str, edge: tuple[str, str]) -> bool:
        """Verify if an edge should exist by prompting the LLM back."""
        verification_prompt = self.prompts["verification_prompts"]["edge_existence"].format(
            premise=premise,
            edge_from=edge[0],
            edge_to=edge[1]
        )
        system_prompt = self.prompts["system_prompts"]["edge_verification"]

        response = self.llm.generate(verification_prompt, system_prompt)[0]
        logging.info(f"Edge verification ({edge[0]}-{edge[1]}): {response}")
        return "YES" in response.upper()
