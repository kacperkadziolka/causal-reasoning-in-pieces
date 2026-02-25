import json
import logging
import os
import re

import pandas as pd
import yaml


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


PROMPTS = load_yaml(os.path.join(os.path.dirname(__file__), "prompts.yaml"))
logging.info(f"Loaded prompts from prompts.yaml: {list(PROMPTS.keys())}")


def load_dataset(file_path: str) -> list[dict]:
    """
    Load the NLGraph shortest path dataset from JSON.

    Handles two formats:
    - List of dicts (enriched format from our notebook)
    - Dict of dicts (raw NLGraph main.json format: {"0": {...}, "1": {...}})

    Adds 'input' key as alias for 'question' (self-planned compatibility).
    """
    with open(file_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        samples = []
        for idx, entry in raw.items():
            sample = {"id": int(idx), **entry}
            # Enrich if needed
            if "num_nodes" not in sample:
                sample.update(_parse_graph_metadata(sample["question"], sample["answer"]))
            samples.append(sample)
        samples.sort(key=lambda x: x["id"])
    elif isinstance(raw, list):
        samples = raw
    else:
        raise ValueError(f"Unexpected JSON format: {type(raw)}")

    # Add 'input' alias for self-planned compatibility
    for sample in samples:
        sample["input"] = sample["question"]

    logging.info(f"Loaded {len(samples)} samples from {file_path}")
    return samples


def _parse_graph_metadata(question: str, answer: str) -> dict:
    """Parse num_nodes, num_edges, path_length, total_weight from question/answer text."""
    node_match = re.search(r"numbered from (\d+) to (\d+)", question)
    num_nodes = int(node_match.group(2)) + 1 if node_match else 0

    edges = re.findall(r"edge between node (\d+) and node (\d+) with weight (\d+)", question)
    num_edges = len(edges)

    path_match = re.search(r"is ([\d,]+) with a total weight of (\d+)", answer)
    if path_match:
        path = [int(n) for n in path_match.group(1).split(",")]
        total_weight = int(path_match.group(2))
    else:
        path = []
        total_weight = None

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "path_length": len(path),
        "total_weight": total_weight,
    }


def _ensure_ground_truth_weight(sample: dict) -> int:
    """Extract ground truth weight, falling back to answer text parsing."""
    weight = sample.get("total_weight")
    if weight is not None:
        return int(weight)

    # Fallback: parse from answer text
    match = re.search(r"total weight of (\d+)", sample.get("answer", ""))
    if match:
        return int(match.group(1))

    raise ValueError(f"Cannot determine ground truth weight for sample {sample.get('id')}")


def prepare_experiment_from_sample(sample: dict, prompt_type: str = "direct_prompt") -> dict:
    """
    Create an experiment dictionary from a dataset sample.

    Parameters:
    - sample: dict with keys: id, question, answer, difficulty, num_nodes, num_edges, total_weight
    - prompt_type: Which prompt template to use
    """
    if prompt_type == "bag_prompt":
        # Build-a-Graph: insert instruction before "Q:" line, matching the paper exactly
        question = sample["question"]
        q_pos = question.rfind("\nQ:")
        question_with_bag = (
            question[:q_pos]
            + "\nLet's construct a graph with the nodes and edges first.\n"
            + question[q_pos:]
        )
        prompt = PROMPTS[prompt_type].format(question_with_bag=question_with_bag)
    else:
        prompt = PROMPTS[prompt_type].format(question=sample["question"])

    return {
        "sample_id": sample["id"],
        "question": sample["question"],
        "ground_truth_answer": sample["answer"],
        "ground_truth_weight": _ensure_ground_truth_weight(sample),
        "difficulty": sample["difficulty"],
        "num_nodes": sample.get("num_nodes"),
        "num_edges": sample.get("num_edges"),
        "prompt": prompt,
        "model_answer": None,
        "model_path": None,
        "model_weight": None,
        "attempt_count": 0,
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def append_log(log_file: str, log_entry: dict) -> None:
    """Append a single log entry to a CSV file."""
    log_df = pd.DataFrame([log_entry])

    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False, mode="w")
    else:
        log_df.to_csv(log_file, index=False, header=False, mode="a")
