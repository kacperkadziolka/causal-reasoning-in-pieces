import logging
from abc import ABC, abstractmethod
from typing import Any

from causal_discovery.llm_client import BaseLLMClient
from causal_discovery.utils import load_prompts, extract_causal_skeleton_json, extract_v_structures_json, \
    extract_directed_edges_literal_format_json, extract_hypothesis_answer, extract_undirected_edges_literal_format_json


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

    def _update_token_usage(self, sample: dict[str, Any], usage: dict) -> None:
        if "token_usage" not in sample:
            sample["token_usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "per_stage": {}
            }

        sample["token_usage"]["input_tokens"] += usage.prompt_tokens
        sample["token_usage"]["output_tokens"] += usage.completion_tokens
        sample["token_usage"]["total_tokens"] += usage.total_tokens

        stage_dict = sample["token_usage"]["per_stage"].setdefault(
            self.__class__.__name__,
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )
        stage_dict["input_tokens"] += usage.prompt_tokens
        stage_dict["output_tokens"] += usage.completion_tokens
        stage_dict["total_tokens"] += usage.total_tokens

        logging.info(
            f"[{self.__class__.__name__}]   total so far: {sample['token_usage']['per_stage'][self.__class__.__name__]}"
        )
        logging.info(f"Overall token usage: {sample['token_usage']}")


class UndirectedSkeletonStage(Stage):
    """
    Stage for generating the undirected skeleton of the causal graph.
    """
    prompt_template = Stage.prompts["undirected_skeleton"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        if "premise" not in input_data:
            raise ValueError("Input data must contain Premise.")

        # 2. Build prompt
        prompt = self.prompt_template.format(premise=input_data["premise"])

        # 3. Send request to LLM
        logging.info("UndirectedSkeletonStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            skeleton = extract_causal_skeleton_json(answer=response)
            input_data["nodes"] = skeleton["nodes"]
            input_data["undirected_edges"] = skeleton["edges"]
        except Exception as e:
            logging.error("Error extracting skeleton: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["nodes"] = None
            input_data["undirected_edges"] = None

        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("UndirectedSkeletonStage: Processing batch with %d samples.", len(inputs))
        # 1. Validate inputs
        for i, input_data in enumerate(inputs):
            if "premise" not in input_data:
                logging.error("Sample %d is missing 'premise' key.", i)
                raise ValueError(f"Sample {i} must contain 'premise'.")
            else:
                logging.debug("Sample %d contains 'premise'.", i)

        # 2. Build prompts
        prompts = []
        for i, input_data in enumerate(inputs):
            try:
                prompt = self.prompt_template.format(premise=input_data["premise"])
                prompts.append(prompt)
                logging.debug("Constructed prompt for sample %d: %s", i, prompt)
            except Exception as e:
                logging.error("Error constructing prompt for sample %d: %s", i, e)
                raise

        logging.debug("All prompts constructed: %s", prompts)

        # 3. Send batch
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

        # 4. Unpack responses into texts and usages, and update token usage
        for i, ((text, usage), item) in enumerate(zip(responses, inputs)):
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(item, usage)

        # 5. Parse skeleton from each response text
        for i, ((text, _), item) in enumerate(zip(responses, inputs)):
            try:
                skeleton = extract_causal_skeleton_json(answer=text)
                item["nodes"] = skeleton["nodes"]
                item["undirected_edges"] = skeleton["edges"]
                logging.debug("Extracted skeleton for sample %d: nodes: %s, edges: %s",
                              i, skeleton["nodes"], skeleton["edges"])
            except Exception as e:
                logging.error("Error extracting skeleton for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                item["nodes"] = None
                item["undirected_edges"] = None

        return inputs


class VStructuresStage(Stage):
    """
    Stage for generating the V-structures out of the causal graph and Premise.
    """
    prompt_template = Stage.prompts["v_structures"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Input data must contain: {', '.join(required_keys)}.")

        # 2. Build prompt
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=self._format_edges(input_data["undirected_edges"]),
        )

        # 3. Send request to LLM
        logging.info("VStructuresStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            v_structures = extract_v_structures_json(answer=response)
            input_data["v_structures"] = v_structures
        except Exception as e:
            logging.error("Error extracting V-structures: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["v_structures"] = None

        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("VStructuresStage: Processing batch with %d samples.", len(inputs))

        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges"}
        for i, input_data in enumerate(inputs):
            if not required_keys.issubset(input_data):
                missing = required_keys - input_data.keys()
                logging.error("Sample %d is missing keys: %s", i, missing)
                raise ValueError(f"Sample {i} must contain: {', '.join(required_keys)}.")
            else:
                logging.debug("Sample %d contains all required keys.", i)

        # 2. Build prompts
        prompts = []
        for i, input_data in enumerate(inputs):
            try:
                prompt = self.prompt_template.format(
                    premise=input_data["premise"],
                    nodes=input_data["nodes"],
                    edges=self._format_edges(input_data["undirected_edges"]),
                )
                prompts.append(prompt)
                logging.debug("Constructed prompt for sample %d: %s", i, prompt)
            except Exception as e:
                logging.error("Error constructing prompt for sample %d: %s", i, e)
                raise

        logging.debug("All prompts constructed: %s", prompts)

        # 3. Send batch
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

        # 4. Unpack responses into texts and usages, and update token usage
        for i, ((text, usage), item) in enumerate(zip(responses, inputs)):
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(item, usage)

        # 5. Parse skeleton from each response text
        for i, ((text, _), item) in enumerate(zip(responses, inputs)):
            try:
                v_structures = extract_v_structures_json(answer=text)
                item["v_structures"] = v_structures
                logging.debug("Extracted V-structures for sample %d: %s", i, v_structures)
            except Exception as e:
                logging.error("Error extracting V-structures for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                item["v_structures"] = None

        return inputs

class MeekRulesStage(Stage):
    """
    Stage for applying Meek's rules to the V-structures.
    """
    prompt_template = Stage.prompts["meek_rules"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges", "v_structures"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Meek rules stage input data must contain: {', '.join(required_keys)}.")

        # 2. Build prompt
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=self._format_edges(input_data["undirected_edges"]),
            v_structures=input_data["v_structures"])

        # 3. Send request to LLM
        logging.info("MeekRulesStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            directed_edges = extract_directed_edges_literal_format_json(answer=response)
            undirected_edges = extract_undirected_edges_literal_format_json(answer=response)
            input_data["directed_edges"] = directed_edges
            input_data["undirected_edges"] = undirected_edges
        except Exception as e:
            logging.error("Error extracting directed edges: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["directed_edges"] = None
            input_data["undirected_edges"] = None
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("MeekRulesStage: Processing batch with %d samples.", len(inputs))

        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges", "v_structures"}
        for i, input_data in enumerate(inputs):
            missing_keys = required_keys - input_data.keys()
            if missing_keys:
                logging.error("Sample %d is missing keys: %s", i, missing_keys)
                raise ValueError(f"Sample {i}: Input data must contain: {', '.join(required_keys)}.")
            else:
                logging.debug("Sample %d contains all required keys.", i)

        # 2. Build prompts
        prompts = []
        for i, input_data in enumerate(inputs):
            try:
                prompt = self.prompt_template.format(
                    premise=input_data["premise"],
                    nodes=input_data["nodes"],
                    edges=self._format_edges(input_data["undirected_edges"]),
                    v_structures=input_data["v_structures"]
                )
                prompts.append(prompt)
                logging.debug("Constructed prompt for sample %d: %s", i, prompt)
            except Exception as e:
                logging.error("Error constructing prompt for sample %d: %s", i, e)
                raise

        logging.debug("All prompts constructed for batch: %s", prompts)

        # 3. Send batch
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

            # 4. Unpack responses into texts and usages, and update token usage
        for i, ((text, usage), item) in enumerate(zip(responses, inputs)):
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(item, usage)

        for i, ((text, _), item) in enumerate(zip(responses, inputs)):
            try:
                directed_edges = extract_directed_edges_literal_format_json(answer=text)
                undirected_edges = extract_undirected_edges_literal_format_json(answer=text)
                logging.debug("Extracted directed_edges for sample %d: %s", i, directed_edges)
            except Exception as e:
                logging.error("Error extracting directed_edges for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                directed_edges = None
                undirected_edges = None
            item["directed_edges"] = directed_edges
            item["undirected_edges"] = undirected_edges

        return inputs


class HypothesisEvaluationStage(Stage):
    """
    Stage for evaluating the hypothesis based on the directed edges.
    """
    prompt_template = Stage.prompts["hypothesis_evaluation"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "directed_edges", "hypothesis", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Hypothesis evaluation stage input data must contain: {', '.join(required_keys)}.")

        # 2. Build prompt
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            directed_edges=self._format_edges(input_data["directed_edges"]),
            undirected_edges=self._format_edges(input_data["undirected_edges"]),
            hypothesis=input_data["hypothesis"]
        )

        # 3. Send request to LLM
        logging.info("HypothesisEvaluationStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 4. Unpack responses and update token usage
        self._update_token_usage(input_data, usage)
        try:
            hypothesis_label = extract_hypothesis_answer(answer=response)
            input_data["hypothesis_label"] = hypothesis_label
        except Exception as e:
            logging.error("Error extracting hypothesis_label: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["hypothesis_label"] = None
        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("HypothesisEvaluationStage: Processing batch with %d samples.", len(inputs))

        # 1. Validate inputs
        required_keys = {"premise", "nodes", "directed_edges", "hypothesis", "undirected_edges"}
        for i, input_data in enumerate(inputs):
            if not required_keys.issubset(input_data):
                missing = required_keys - input_data.keys()
                logging.error("Sample %d is missing keys: %s", i, missing)
                raise ValueError(f"Sample {i} must contain: {', '.join(required_keys)}.")
            else:
                logging.debug("Sample %d contains all required keys.", i)

        # 2. Build prompts
        prompts = []
        for i, input_data in enumerate(inputs):
            try:
                prompt = self.prompt_template.format(
                    premise=input_data["premise"],
                    nodes=input_data["nodes"],
                    directed_edges=self._format_edges(input_data["directed_edges"]),
                    undirected_edges=self._format_edges(input_data["undirected_edges"]),
                    hypothesis=input_data["hypothesis"]
                )
                prompts.append(prompt)
                logging.debug("Constructed prompt for sample %d: %s", i, prompt)
            except Exception as e:
                logging.error("Error constructing prompt for sample %d: %s", i, e)
                raise

        logging.debug("All prompts constructed: %s", prompts)

        # 3. Send batch
        try:
            responses = self.client.complete_batch(prompts=prompts)
            logging.info("Batch call returned %d responses.", len(responses))
        except Exception as e:
            logging.error("Batch call failed: %s", e)
            raise

        # 4. Unpack responses into texts and usages, and update token usage
        for i, ((text, usage), item) in enumerate(zip(responses, inputs)):
            logging.debug("Raw response text for sample %d: %s", i, text)
            logging.debug("Token usage for sample %d: %s", i, usage)
            self._update_token_usage(item, usage)

        for i, ((text, _), item) in enumerate(zip(responses, inputs)):
            try:
                hypothesis_label = extract_hypothesis_answer(answer=text)
                item["hypothesis_label"] = hypothesis_label
                logging.debug("Extracted hypothesis_label for sample %d: %s", i, hypothesis_label)
            except Exception as e:
                logging.error("Error extracting hypothesis_label for sample %d: %s", i, e)
                logging.debug("Problematic response for sample %d: %s", i, text)
                item["hypothesis_label"] = None

        return inputs
