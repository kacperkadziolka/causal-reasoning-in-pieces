import json
import re
from functools import lru_cache
from typing import Any

import yaml


def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    """
    Splits the list 'lst' into chunks of size 'chunk_size'.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


@lru_cache(maxsize=1)
def load_prompts(file_path: str = "prompts.yaml") -> dict[str, str]:
    """
    Load prompts from a YAML file.

    Parameters:
    - file_path (str): Path to the YAML file containing prompts.

    Returns:
    - dict: A dictionary where keys are prompt names and values are the corresponding prompts.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def extract_premise(text: str) -> str:
    """
    Extracts Premise from the input text.

    Parameters:
    - text (str): The input text containing "Premise:" and possibly "Hypothesis:".

    Returns:
    - str: The extracted premise.
    """
    # Use regex to extract text between "Premise:" and "Hypothesis:"
    match = re.search(r"Premise:\s*(.*?)\s*Hypothesis:", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If "Hypothesis:" is not present, extract everything after "Premise:"
        match = re.search(r"Premise:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            raise RuntimeError("Failed to extract premise from the text.")


def extract_hypothesis(text: str) -> str:
    """
    Extracts Hypothesis from the input text.

    Parameters:
    - text (str): The input text containing "Hypothesis:".

    Returns:
    - str: The extracted hypothesis.
    """
    # Use regex to extract everything after "Hypothesis:"
    match = re.search(r"Hypothesis:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        raise RuntimeError("Failed to extract hypothesis from the text.")


def extract_causal_skeleton_json(answer: str) -> dict[str, Any]:
    """
    Extract the causal skeleton (nodes and edges) from the provided LLM answer string using the JSON format.

    :param answer: The answer returned by the LLM API.
    :return: A dictionary containing the nodes (as a list) and edges (as a set of sorted tuples).
    """
    try:
        # First approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)

            result = {}
            # Extract nodes
            if "nodes" in data:
                result["nodes"] = data["nodes"]
            else:
                raise ValueError("No nodes found in the JSON data in the extract casual skeleton stage.")

            # Extract edges and convert to set of sorted tuples
            if "edges" in data:
                result["edges"] = set(tuple(sorted(edge)) for edge in data["edges"])
            else:
                raise ValueError("No edges found in the extract casual skeleton stage.")

            return result

        # Second approach: Find all potential JSON objects and test each one
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "nodes" in data and "edges" in data:
                    result = {
                        "nodes": data["nodes"],
                        "edges": set(tuple(sorted(edge)) for edge in data["edges"])
                    }
                    return result
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON with nodes and edges found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract causal skeleton: {e}")


def extract_v_structures_json(answer: str) -> list[tuple]:
    """
    Extract the v-structures from the provided LLM answer string using the JSON format.

    :param answer: The answer returned by the LLM API.
    :return: A list of v-structures extracted from the answer.
    """
    try:
        # First approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)

            # Extract v-structures
            if "v_structures" in data:
                v_structures = [tuple(v_struct) for v_struct in data["v_structures"]]
                return v_structures

        # Second approach: Find all potential JSON objects and test each one
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "v_structures" in data:
                    v_structures = [tuple(v_struct) for v_struct in data["v_structures"]]
                    return v_structures
            except json.JSONDecodeError:
                continue

        # If no v_structures found in JSON, look for v-structures in text format
        v_structures_pattern = r'v.?structures?:?\s*\[(.*?)\]'
        v_struct_match = re.search(v_structures_pattern, answer, re.IGNORECASE | re.DOTALL)
        if v_struct_match:
            content = v_struct_match.group(1)
            # Extract arrays like ["A", "B", "C"]
            array_pattern = r'\[\s*"([A-E])"\s*,\s*"([A-E])"\s*,\s*"([A-E])"\s*\]'
            v_structures = []
            for match in re.finditer(array_pattern, content):
                v_structures.append(tuple(match.groups()))
            if v_structures:
                return v_structures

        raise ValueError("No v-structures found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract v-structures: {e}")


def extract_directed_edges_literal_format_json(answer: str) -> list:
    """
    Extract the directed edges from the provided LLM answer string using the expected JSON format.

    Expected JSON format:
    {
      "final_graph": {
        "directed_edges": [
          {
            "from": "Node1",
            "to": "Node2"
          },
          {
            "from": "Node2",
            "to": "Node3"
          }
        ],
        "undirected_edges": [
          ["Node3", "Node4"],
          ["Node5", "Node6"]
        ]
      }
    }

    :param answer: The answer returned by the LLM API.
    :return: A list of directed edges (tuples representing source->target) extracted from the answer.
    """
    try:
        # First approach: Find JSON block delimited by triple backticks.
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            if "final_graph" in data and "directed_edges" in data["final_graph"]:
                # Return empty list if directed_edges exists but is empty
                if not data["final_graph"]["directed_edges"]:
                    return []

                directed_edges = []
                for edge in data["final_graph"]["directed_edges"]:
                    if "from" in edge and "to" in edge:
                        directed_edges.append((edge["from"], edge["to"]))
                return directed_edges

        # Second approach: Search for any JSON objects within the answer.
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "final_graph" in data and "directed_edges" in data["final_graph"]:
                    # Return empty list if directed_edges exists but is empty
                    if not data["final_graph"]["directed_edges"]:
                        return []

                    directed_edges = []
                    for edge in data["final_graph"]["directed_edges"]:
                        if "from" in edge and "to" in edge:
                            directed_edges.append((edge["from"], edge["to"]))
                    return directed_edges
            except json.JSONDecodeError:
                continue

        # Third approach: Look for directed edges in text format from the final_graph section.
        edges_pattern = r'"directed_edges"\s*:\s*\[(.*?)\]'
        edges_match = re.search(edges_pattern, answer, re.IGNORECASE | re.DOTALL)
        if edges_match:
            content = edges_match.group(1).strip()
            # Return empty list if content is empty or only whitespace
            if not content:
                return []

            # Each edge is expected to be in the format {"from": "X", "to": "Y"}
            edge_pattern = r'\{\s*"from"\s*:\s*"([^"]+)"\s*,\s*"to"\s*:\s*"([^"]+)"\s*\}'
            directed_edges = []
            for match in re.finditer(edge_pattern, content):
                directed_edges.append((match.group(1), match.group(2)))
            return directed_edges

        raise ValueError("No directed edges found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract directed edges: {e}")


def extract_hypothesis_answer(answer: str) -> bool:
    """
    Extract the hypothesis answer (True/False) from the provided LLM answer string.

    Parameters:
    - answer (str): The answer returned by the LLM API.

    Returns:
    - bool: The extracted hypothesis answer as a Python boolean.

    Raises:
    - RuntimeError: If extraction fails.
    """
    try:
        # First approach: Try to parse the entire answer as JSON
        try:
            data = json.loads(answer)
            if "hypothesis_answer" in data:
                return bool(data["hypothesis_answer"])
        except json.JSONDecodeError:
            pass

        # Second approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*(.*?)```', answer, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                if "hypothesis_answer" in data:
                    return bool(data["hypothesis_answer"])
            except json.JSONDecodeError:
                pass

        # Third approach: Try to find complete JSON objects in the text
        potential_json_start = 0
        while potential_json_start < len(answer):
            start_idx = answer.find('{', potential_json_start)
            if start_idx == -1:
                break

            # Find matching closing brace by tracking brace balance
            open_braces = 1
            for end_idx in range(start_idx + 1, len(answer)):
                if answer[end_idx] == '{':
                    open_braces += 1
                elif answer[end_idx] == '}':
                    open_braces -= 1

                if open_braces == 0:
                    # Found a potential JSON object
                    try:
                        json_str = answer[start_idx:end_idx + 1]
                        data = json.loads(json_str)
                        if "hypothesis_answer" in data:
                            return bool(data["hypothesis_answer"])
                    except json.JSONDecodeError:
                        pass
                    break

            potential_json_start = start_idx + 1

        raise ValueError("No valid JSON with hypothesis_answer found")
    except Exception as e:
        raise RuntimeError(f"Failed to extract hypothesis answer: {e}")
