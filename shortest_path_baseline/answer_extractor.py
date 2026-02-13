import json
import re
from typing import Optional


def extract_shortest_path_answer(answer: str) -> dict:
    """
    Extract path and weight from the model's free-text response.

    Uses multiple extraction strategies in order of strictness:
    1. Exact NLGraph format: "shortest path ... is A,B,C with a total weight of W"
    2. Separate path + weight patterns (looser)
    3. JSON object: {"path": [...], "weight": N}
    4. Fallback: last comma-separated number sequence + nearby weight number

    NOTE: All strategies use the LAST match to avoid extracting from echoed
    question text (the model may repeat the question before answering).

    Returns:
        {"path": list[int] | None, "weight": int | None}
    """
    # Strategy 1: Exact NLGraph answer format — take the LAST match
    matches = list(re.finditer(
        r"shortest path.*?is\s+(\d+(?:\s*,\s*\d+)*)\s+with\s+a\s+total\s+weight\s+of\s+(\d+)",
        answer,
        re.IGNORECASE | re.DOTALL,
    ))
    if matches:
        match = matches[-1]
        path = [int(n.strip()) for n in match.group(1).split(",")]
        weight = int(match.group(2))
        return {"path": path, "weight": weight}

    # Strategy 2: Separate path and weight patterns — take the LAST match
    path_matches = list(re.finditer(
        r"(?:path|route)[\s:]*(?:is|=)?\s*(\d+(?:\s*,\s*\d+)*)",
        answer,
        re.IGNORECASE,
    ))
    path_match = path_matches[-1] if path_matches else None
    weight_matches = list(re.finditer(
        r"(?:total\s+)?weight[\s:]*(?:is|of|=)?\s*(\d+)",
        answer,
        re.IGNORECASE,
    ))
    weight_match = weight_matches[-1] if weight_matches else None
    if path_match and weight_match:
        path = [int(n.strip()) for n in path_match.group(1).split(",")]
        weight = int(weight_match.group(1))
        return {"path": path, "weight": weight}

    # Strategy 3: JSON object
    try:
        data = json.loads(answer)
        if isinstance(data, dict) and "path" in data and "weight" in data:
            return {"path": [int(n) for n in data["path"]], "weight": int(data["weight"])}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Also try to find JSON within the text
    json_match = re.search(r"\{[^}]*\"path\"[^}]*\}", answer)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if "path" in data and "weight" in data:
                return {"path": [int(n) for n in data["path"]], "weight": int(data["weight"])}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 4: Fallback — find comma-separated number sequences and standalone weight
    all_sequences = re.findall(r"(\d+(?:\s*,\s*\d+)+)", answer)
    if all_sequences and weight_match:
        # Take the longest sequence as the likely path
        longest = max(all_sequences, key=lambda s: len(s.split(",")))
        path = [int(n.strip()) for n in longest.split(",")]
        weight = int(weight_match.group(1))
        return {"path": path, "weight": weight}

    # If we only have a weight match but no path
    if weight_match:
        return {"path": None, "weight": int(weight_match.group(1))}

    # If we only have a path match but no weight
    if path_match:
        path = [int(n.strip()) for n in path_match.group(1).split(",")]
        return {"path": path, "weight": None}

    return {"path": None, "weight": None}


def parse_graph_from_question(question: str) -> dict:
    """
    Parse graph structure from the NLGraph question text.

    Returns:
        {
            "num_nodes": int,
            "edges": list of (node1, node2, weight) tuples,
            "source": int,
            "target": int,
        }
    """
    result = {"num_nodes": 0, "edges": [], "source": None, "target": None}

    node_match = re.search(r"numbered from (\d+) to (\d+)", question)
    if node_match:
        result["num_nodes"] = int(node_match.group(2)) + 1

    edge_pattern = r"edge between node (\d+) and node (\d+) with weight (\d+)"
    for match in re.finditer(edge_pattern, question):
        node1, node2, weight = int(match.group(1)), int(match.group(2)), int(match.group(3))
        result["edges"].append((node1, node2, weight))

    query_match = re.search(r"shortest path from node (\d+) to node (\d+)", question)
    if query_match:
        result["source"] = int(query_match.group(1))
        result["target"] = int(query_match.group(2))

    return result


def validate_path(
    path: list[int],
    edges: list[tuple[int, int, int]],
    source: int,
    target: int,
) -> tuple[bool, Optional[int]]:
    """
    Validate that a path is valid in the graph.

    Checks:
    - Path starts at source and ends at target
    - All consecutive edges exist in the graph
    - Computes the actual path weight

    Returns:
        (is_valid, computed_weight) — computed_weight is None if path is invalid
    """
    if not path or len(path) < 2:
        return False, None

    if path[0] != source or path[-1] != target:
        return False, None

    # Build adjacency lookup
    adj = {}
    for u, v, w in edges:
        adj[(u, v)] = w
        adj[(v, u)] = w  # undirected

    total_weight = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        if edge not in adj:
            return False, None
        total_weight += adj[edge]

    return True, total_weight


def compare_shortest_path(
    ground_truth_weight: int,
    model_path: Optional[list[int]],
    model_weight: Optional[int],
    graph_info: Optional[dict] = None,
) -> dict:
    """
    Compare model answer against ground truth.

    Parameters:
    - ground_truth_weight: the correct shortest path weight
    - model_path: extracted path from model response (may be None)
    - model_weight: extracted weight from model response (may be None)
    - graph_info: parsed graph info for path validation (optional)

    Returns:
        {
            "weight_correct": bool,
            "path_valid": bool or None (if no graph_info),
            "path_weight_consistent": bool or None,
            "extraction_failed": bool,
        }
    """
    extraction_failed = model_weight is None
    weight_correct = (model_weight == ground_truth_weight) if model_weight is not None else False

    path_valid = None
    path_weight_consistent = None

    if graph_info and model_path:
        is_valid, computed_weight = validate_path(
            model_path,
            graph_info["edges"],
            graph_info["source"],
            graph_info["target"],
        )
        path_valid = is_valid
        if is_valid and computed_weight is not None and model_weight is not None:
            path_weight_consistent = computed_weight == model_weight

    return {
        "weight_correct": weight_correct,
        "path_valid": path_valid,
        "path_weight_consistent": path_weight_consistent,
        "extraction_failed": extraction_failed,
    }


def aggregate_metrics(results: list[dict], difficulties: list[str]) -> dict:
    """
    Aggregate metrics from multiple experiments.

    Parameters:
    - results: list of compare_shortest_path() outputs
    - difficulties: corresponding difficulty labels

    Returns:
        dict with overall and per-difficulty metrics
    """
    if not results:
        return {"error": "No results to aggregate"}

    total = len(results)
    extraction_failures = sum(1 for r in results if r["extraction_failed"])
    weight_correct = sum(1 for r in results if r["weight_correct"])

    valid_paths = [r for r in results if r["path_valid"] is not None]
    path_valid_count = sum(1 for r in valid_paths if r["path_valid"])

    metrics = {
        "total": total,
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failures / total if total > 0 else 0,
        "weight_accuracy": weight_correct / total if total > 0 else 0,
        "path_valid_rate": path_valid_count / len(valid_paths) if valid_paths else None,
    }

    # Breakdown by difficulty
    by_difficulty = {}
    for diff in ["easy", "hard"]:
        indices = [i for i, d in enumerate(difficulties) if d == diff]
        if not indices:
            continue
        diff_results = [results[i] for i in indices]
        diff_total = len(diff_results)
        diff_failures = sum(1 for r in diff_results if r["extraction_failed"])
        diff_correct = sum(1 for r in diff_results if r["weight_correct"])

        by_difficulty[diff] = {
            "total": diff_total,
            "extraction_failures": diff_failures,
            "weight_accuracy": diff_correct / diff_total if diff_total > 0 else 0,
        }

    metrics["by_difficulty"] = by_difficulty
    return metrics


def display_metrics(metrics: dict) -> None:
    """Display metrics in a user-friendly format."""
    print("\n" + "=" * 50)
    print("Aggregate Metrics")
    print("=" * 50)

    print(f"Total samples: {metrics['total']}")
    print(f"Extraction failures: {metrics['extraction_failures']} ({metrics['extraction_failure_rate']:.1%})")
    print(f"Weight accuracy: {metrics['weight_accuracy']:.2%}")

    if metrics.get("path_valid_rate") is not None:
        print(f"Path validity rate: {metrics['path_valid_rate']:.2%}")

    if metrics.get("by_difficulty"):
        print("\nBy Difficulty:")
        print("-" * 40)
        for diff, diff_metrics in metrics["by_difficulty"].items():
            print(
                f"  {diff.capitalize():>5}: "
                f"{diff_metrics['weight_accuracy']:.2%} accuracy "
                f"({diff_metrics['total']} samples, "
                f"{diff_metrics['extraction_failures']} extraction failures)"
            )

    print("=" * 50)
