import logging
from typing import Any

from causal_discovery.pipeline.stages import Stage as CausalStage
from shortest_path.utils import (
    load_prompts,
    extract_graph_json,
    extract_path_weight_json,
    extract_verified_result_json,
)


class Stage(CausalStage):
    """Base class for shortest path pipeline stages.
    Inherits from causal_discovery Stage but loads shortest_path prompts."""
    prompts: dict[str, str] = load_prompts()

    def _update_token_usage(self, sample: dict[str, Any], usage) -> None:
        if usage is None:
            return
        super()._update_token_usage(sample, usage)

    def _has_valid_inputs(self, input_data: dict[str, Any], required_keys: set[str]) -> bool:
        """Check that required keys exist and are not None."""
        return all(input_data.get(k) is not None for k in required_keys)


class GraphParsingStage(Stage):
    """Stage 1: Parse natural language graph description into structured JSON."""
    prompt_template = Stage.prompts["graph_parsing"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        if "question" not in input_data:
            raise ValueError("Input data must contain 'question'.")

        prompt = self.prompt_template.format(question=input_data["question"])

        logging.info("GraphParsingStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        self._update_token_usage(input_data, usage)
        try:
            graph = extract_graph_json(answer=response)
            input_data["nodes"] = graph["nodes"]
            input_data["adjacency_list"] = graph["adjacency_list"]
            input_data["source"] = graph["source"]
            input_data["target"] = graph["target"]
        except Exception as e:
            logging.error("Error extracting graph: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["nodes"] = None
            input_data["adjacency_list"] = None
            input_data["source"] = None
            input_data["target"] = None

        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("GraphParsingStage: Processing batch with %d samples.", len(inputs))

        for i, item in enumerate(inputs):
            if "question" not in item:
                raise ValueError(f"Sample {i} must contain 'question'.")

        prompts = [self.prompt_template.format(question=item["question"]) for item in inputs]

        responses = self.client.complete_batch(prompts=prompts)

        for i, ((text, usage), item) in enumerate(zip(responses, inputs)):
            self._update_token_usage(item, usage)

        for i, ((text, _), item) in enumerate(zip(responses, inputs)):
            try:
                graph = extract_graph_json(answer=text)
                item["nodes"] = graph["nodes"]
                item["adjacency_list"] = graph["adjacency_list"]
                item["source"] = graph["source"]
                item["target"] = graph["target"]
            except Exception as e:
                logging.error("Error extracting graph for sample %d: %s", i, e)
                item["nodes"] = None
                item["adjacency_list"] = None
                item["source"] = None
                item["target"] = None

        return inputs


class DijkstraExecutionStage(Stage):
    """Stage 2: Execute Dijkstra's algorithm on the parsed graph."""
    prompt_template = Stage.prompts["dijkstra_execution"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        required_keys = {"nodes", "adjacency_list", "source", "target"}
        if not self._has_valid_inputs(input_data, required_keys):
            logging.warning("DijkstraExecutionStage: Skipping — previous stage produced None values.")
            input_data["path"] = None
            input_data["total_weight"] = None
            return input_data

        prompt = self.prompt_template.format(
            nodes=input_data["nodes"],
            adjacency_list=input_data["adjacency_list"],
            source=input_data["source"],
            target=input_data["target"],
        )

        logging.info("DijkstraExecutionStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        self._update_token_usage(input_data, usage)
        try:
            result = extract_path_weight_json(answer=response)
            input_data["path"] = result["path"]
            input_data["total_weight"] = result["total_weight"]
        except Exception as e:
            logging.error("Error extracting path/weight: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["path"] = None
            input_data["total_weight"] = None

        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("DijkstraExecutionStage: Processing batch with %d samples.", len(inputs))

        required_keys = {"nodes", "adjacency_list", "source", "target"}

        # Separate valid and failed samples
        valid_indices = []
        for i, item in enumerate(inputs):
            if self._has_valid_inputs(item, required_keys):
                valid_indices.append(i)
            else:
                logging.warning("DijkstraExecutionStage: Sample %d has None values, skipping.", i)
                item["path"] = None
                item["total_weight"] = None

        if not valid_indices:
            return inputs

        valid_items = [inputs[i] for i in valid_indices]
        prompts = [
            self.prompt_template.format(
                nodes=item["nodes"],
                adjacency_list=item["adjacency_list"],
                source=item["source"],
                target=item["target"],
            )
            for item in valid_items
        ]

        responses = self.client.complete_batch(prompts=prompts)

        for (text, usage), item in zip(responses, valid_items):
            self._update_token_usage(item, usage)

        for (text, _), item in zip(responses, valid_items):
            try:
                result = extract_path_weight_json(answer=text)
                item["path"] = result["path"]
                item["total_weight"] = result["total_weight"]
            except Exception as e:
                logging.error("Error extracting path/weight: %s", e)
                item["path"] = None
                item["total_weight"] = None

        return inputs


class ResultVerificationStage(Stage):
    """Stage 3 (optional): Verify and potentially correct the shortest path result."""
    prompt_template = Stage.prompts["result_verification"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        required_keys = {"nodes", "adjacency_list", "source", "target", "path", "total_weight"}
        if not self._has_valid_inputs(input_data, required_keys):
            logging.warning("ResultVerificationStage: Skipping — previous stage produced None values.")
            return input_data

        prompt = self.prompt_template.format(
            nodes=input_data["nodes"],
            adjacency_list=input_data["adjacency_list"],
            source=input_data["source"],
            target=input_data["target"],
            path=input_data["path"],
            total_weight=input_data["total_weight"],
        )

        logging.info("ResultVerificationStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        self._update_token_usage(input_data, usage)
        try:
            result = extract_verified_result_json(answer=response)
            input_data["path"] = result["path"]
            input_data["total_weight"] = result["total_weight"]
        except Exception as e:
            logging.error("Error extracting verified result: %s", e)
            logging.debug("Problematic response: %s", response)

        return input_data

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("ResultVerificationStage: Processing batch with %d samples.", len(inputs))

        required_keys = {"nodes", "adjacency_list", "source", "target", "path", "total_weight"}

        valid_indices = []
        for i, item in enumerate(inputs):
            if self._has_valid_inputs(item, required_keys):
                valid_indices.append(i)
            else:
                logging.warning("ResultVerificationStage: Sample %d has None values, skipping.", i)

        if not valid_indices:
            return inputs

        valid_items = [inputs[i] for i in valid_indices]
        prompts = [
            self.prompt_template.format(
                nodes=item["nodes"],
                adjacency_list=item["adjacency_list"],
                source=item["source"],
                target=item["target"],
                path=item["path"],
                total_weight=item["total_weight"],
            )
            for item in valid_items
        ]

        responses = self.client.complete_batch(prompts=prompts)

        for (text, usage), item in zip(responses, valid_items):
            self._update_token_usage(item, usage)

        for (text, _), item in zip(responses, valid_items):
            try:
                result = extract_verified_result_json(answer=text)
                item["path"] = result["path"]
                item["total_weight"] = result["total_weight"]
            except Exception as e:
                logging.error("Error extracting verified result: %s", e)

        return inputs
