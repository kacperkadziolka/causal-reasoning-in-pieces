import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_prompts(file_path: str = "prompts.yaml") -> dict[str, str]:
    if not Path(file_path).is_absolute():
        file_path = Path(__file__).parent / file_path
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset(path: str) -> list[dict]:
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        samples = list(data.values())
    else:
        samples = data

    logging.info(f"Loaded {len(samples)} samples from {path}")
    return samples


def _find_json_in_response(answer: str) -> list[dict]:
    """Extract JSON objects from LLM response using multiple strategies."""
    results = []

    # Strategy 1: JSON in triple-backtick block
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', answer, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            results.append(data)
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find JSON objects by brace matching
    i = 0
    while i < len(answer):
        start = answer.find('{', i)
        if start == -1:
            break
        depth = 0
        for end in range(start, len(answer)):
            if answer[end] == '{':
                depth += 1
            elif answer[end] == '}':
                depth -= 1
            if depth == 0:
                try:
                    data = json.loads(answer[start:end + 1])
                    results.append(data)
                except json.JSONDecodeError:
                    pass
                break
        i = start + 1

    return results


def extract_graph_json(answer: str) -> dict[str, Any]:
    """Extract parsed graph structure from GraphParsingStage LLM response."""
    if answer is None:
        raise RuntimeError("LLM returned no response (None).")
    candidates = _find_json_in_response(answer)

    for data in candidates:
        if "nodes" in data and "adjacency_list" in data:
            source = data.get("source")
            target = data.get("target")

            # Ensure source/target are ints
            if source is not None:
                source = int(source)
            if target is not None:
                target = int(target)

            # Ensure nodes are ints
            nodes = [int(n) for n in data["nodes"]]

            return {
                "nodes": nodes,
                "adjacency_list": data["adjacency_list"],
                "source": source,
                "target": target,
                "num_nodes": data.get("num_nodes", len(nodes)),
                "num_edges": data.get("num_edges"),
            }

    raise RuntimeError(f"Failed to extract graph JSON from response. No valid JSON with 'nodes' and 'adjacency_list' found.")


def extract_path_weight_json(answer: str) -> dict[str, Any]:
    """Extract path and total_weight from DijkstraExecutionStage or ResultVerificationStage LLM response."""
    if answer is None:
        raise RuntimeError("LLM returned no response (None).")
    candidates = _find_json_in_response(answer)

    # Look for the LAST matching JSON (model may echo earlier data)
    best = None
    for data in candidates:
        if "path" in data and "total_weight" in data:
            best = data

    if best is not None:
        path = best["path"]
        weight = best["total_weight"]

        # Ensure path elements are ints
        if isinstance(path, list):
            try:
                path = [int(n) for n in path]
            except (ValueError, TypeError):
                pass

        # Ensure weight is int
        if weight is not None:
            try:
                weight = int(weight)
            except (ValueError, TypeError):
                pass

        return {"path": path, "total_weight": weight}

    # Fallback: regex for "path": [...] and "total_weight": N
    path_match = re.search(r'"path"\s*:\s*\[([^\]]*)\]', answer)
    weight_match = re.search(r'"total_weight"\s*:\s*(\d+)', answer)

    if path_match and weight_match:
        path_str = path_match.group(1).strip()
        path = [int(x.strip()) for x in path_str.split(',') if x.strip()]
        weight = int(weight_match.group(1))
        return {"path": path, "total_weight": weight}

    raise RuntimeError(f"Failed to extract path/weight from response.")


# Alias — same format
extract_verified_result_json = extract_path_weight_json
