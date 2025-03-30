import logging
from abc import ABC, abstractmethod
from typing import Any

from causal_discovery.llm_client import BaseLLMClient
from causal_discovery.utils import load_prompts, extract_causal_skeleton_json, extract_v_structures_json, \
    extract_directed_edges_literal_format_json, extract_hypothesis_answer


class Stage(ABC):
    """
    Base class for all stages in the pipeline.
    Each subclass needs to implement the `prompt_template` attribute.
    """
    prompts: dict[str, str] = load_prompts()
    prompt_template: str = None

    def __init__(self, client: BaseLLMClient):
        self.client = client
        if self.prompt_template is None:
            raise ValueError("Subclasses must define a prompt_template.")

    @abstractmethod
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process the single sample using the prompt template and stage-specific logic..
        """
        pass

    @abstractmethod
    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process a batch of samples using the prompt template and stage-specific logic.
        """
        pass

    @staticmethod
    def _format_edges(edges: set[tuple]) -> str:
        """
        Format a list of edges with line breaks for better readability in prompts.
        """
        formatted = "[\n    "
        formatted += ",\n    ".join([str(edge) for edge in edges])
        formatted += "\n  ]"
        return formatted


class UndirectedSkeletonStage(Stage):
    """
    Stage for generating the undirected skeleton of the causal graph.
    """
    prompt_template = Stage.prompts["undirected_skeleton"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        if "premise" not in input_data:
            raise ValueError("Input data must contain Premise.")

        prompt = self.prompt_template.format(premise=input_data["premise"])

        logging.info("UndirectedSkeletonStage: Sending prompt to LLM.")
        response = self.client.complete(prompt=prompt)

        skeleton = extract_causal_skeleton_json(answer=response)
        input_data["nodes"] = skeleton["nodes"]
        input_data["undirected_edges"] = skeleton["edges"]
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any("premise" not in input_data for input_data in inputs):
            raise ValueError("All input data items must contain 'premise' key.")

        prompts = [self.prompt_template.format(premise=input_data["premise"]) for input_data in inputs]
        responses = self.client.complete_batch(prompts=prompts)

        for item, response in zip(inputs, responses):
            skeleton = extract_causal_skeleton_json(answer=response)
            item["nodes"] = skeleton["nodes"]
            item["undirected_edges"] = skeleton["edges"]
        return inputs


class VStructuresStage(Stage):
    """
    Stage for generating the V-structures out of the causal graph and Premise.
    """
    prompt_template = Stage.prompts["v_structures"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        required_keys = {"premise", "nodes", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Input data must contain: {', '.join(required_keys)}.")

        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=self._format_edges(input_data["undirected_edges"]),
        )

        logging.info("VStructuresStage: Sending prompt to LLM.")
        response = self.client.complete(prompt=prompt)

        v_structures = extract_v_structures_json(answer=response)
        input_data["v_structures"] = v_structures
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        required_keys = {"premise", "nodes", "undirected_edges"}
        if any(not required_keys.issubset(input_data) for input_data in inputs):
            raise ValueError(f"Input data must contain: {', '.join(required_keys)}.")

        prompts = []
        for input_data in inputs:
            prompts.append(self.prompt_template.format(
                premise=input_data["premise"],
                nodes=input_data["nodes"],
                edges=self._format_edges(input_data["undirected_edges"]),
            ))
        responses = self.client.complete_batch(prompts=prompts)

        for item, response in zip(inputs, responses):
            v_structures = extract_v_structures_json(answer=response)
            item["v_structures"] = v_structures
        return inputs


class MeekRulesStage(Stage):
    """
    Stage for applying Meek's rules to the V-structures.
    """
    prompt_template = Stage.prompts["meek_rules"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        required_keys = {"premise", "nodes", "undirected_edges", "v_structures"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Meek rules stage input data must contain: {', '.join(required_keys)}.")

        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=self._format_edges(input_data["undirected_edges"]),
            v_structures=input_data["v_structures"])

        logging.info("MeekRulesStage: Sending prompt to LLM.")
        response = self.client.complete(prompt=prompt)

        directed_edges = extract_directed_edges_literal_format_json(answer=response)
        input_data["directed_edges"] = directed_edges
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        required_keys = {"premise", "nodes", "undirected_edges", "v_structures"}
        if any(not required_keys.issubset(input_data) for input_data in inputs):
            raise ValueError(f"Input data must contain: {', '.join(required_keys)}.")

        prompts = []
        for input_data in inputs:
            prompts.append(self.prompt_template.format(
                premise=input_data["premise"],
                nodes=input_data["nodes"],
                edges=self._format_edges(input_data["undirected_edges"]),
                v_structures=input_data["v_structures"]
            ))
        responses = self.client.complete_batch(prompts=prompts)

        for item, response in zip(inputs, responses):
            directed_edges = extract_directed_edges_literal_format_json(answer=response)
            item["directed_edges"] = directed_edges
        return inputs


class HypothesisEvaluationStage(Stage):
    """
    Stage for evaluating the hypothesis based on the directed edges.
    """
    prompt_template = Stage.prompts["hypothesis_evaluation"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        required_keys = {"nodes", "directed_edges", "hypothesis"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Hypothesis evaluation stage input data must contain: {', '.join(required_keys)}.")

        prompt = self.prompt_template.format(
            nodes=input_data["nodes"],
            directed_edges=self._format_edges(input_data["directed_edges"]),
            hypothesis=input_data["hypothesis"]
        )

        logging.info("HypothesisEvaluationStage: Sending prompt to LLM.")
        response = self.client.complete(prompt=prompt)

        hypothesis_label = extract_hypothesis_answer(answer=response)
        input_data["hypothesis_label"] = hypothesis_label
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        required_keys = {"nodes", "directed_edges", "hypothesis"}
        if any(not required_keys.issubset(input_data) for input_data in inputs):
            raise ValueError(f"Hypothesis evaluation stage input data must contain: {', '.join(required_keys)}.")

        prompts = []
        for input_data in inputs:
            prompts.append(self.prompt_template.format(
                nodes=input_data["nodes"],
                edges=self._format_edges(input_data["directed_edges"]),
                hypothesis=input_data["hypothesis"]
            ))
        responses = self.client.complete_batch(prompts=prompts)

        for item, response in zip(inputs, responses):
            hypothesis_label = extract_hypothesis_answer(answer=response)
            item["hypothesis_label"] = hypothesis_label
        return inputs
