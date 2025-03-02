import ast
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

    def check_global_consistency(self, premise: str, edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Verify the global consistency of the graph and suggest corrections by prompting the LLM back."""
        formatted_edges = "\n".join([f"- {edge[0]} -- {edge[1]}" for edge in edges])
        consistency_prompt = self.prompts["verification_prompts"]["global_consistency"].format(
            problem_statement=premise,
            formatted_edges=formatted_edges
        )
        system_prompt = self.prompts["system_prompts"]["graph_consistency"]

        print(system_prompt)
        print(consistency_prompt)

        response = self.llm.generate(consistency_prompt, system_prompt)[0]
        logging.info(f"Initial edges: {edges}")
        logging.info(f"Global consistency check: {response}")

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                corrections = ast.literal_eval(json_str)

                current_edges = set(edges)
                for edge in corrections.get("edges_to_add", []):
                    edge_tuple = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                    current_edges.add(edge_tuple)

                for edge in corrections.get("edges_to_remove", []):
                    edge_tuple = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                    if edge_tuple in current_edges:
                        current_edges.remove(edge_tuple)

                logging.info(f"Corrected edges: {list(current_edges)}")
                return list(current_edges)
        except Exception as e:
            logging.error(f"Error parsing consistency check response: {e}")

        # Return original edges if parsing fails
        logging.info(f"Failed to parse consistency check response. Returning original edges.")
        return edges
